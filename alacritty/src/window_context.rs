//! Terminal window context.

use std::error::Error;
use std::fs::File;
use std::io::Write;
use std::mem;
use std::rc::Rc;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Instant;

/// Process-wide monotonically increasing tab ID counter. Ensures that tab IDs are
/// unique across all windows within a process so the Processor's `tab_registry`
/// (keyed by tab ID) never confuses tabs from different windows.
static NEXT_TAB_ID: AtomicUsize = AtomicUsize::new(0);

use glutin::config::Config as GlutinConfig;
use glutin::display::GetGlDisplay;
#[cfg(all(feature = "x11", not(any(target_os = "macos", windows))))]
use glutin::platform::x11::X11GlConfigExt;
use log::{error, info};
use serde_json as json;
use winit::event::{Event as WinitEvent, Modifiers, WindowEvent};
use winit::event_loop::{ActiveEventLoop, EventLoopProxy};
use winit::raw_window_handle::HasDisplayHandle;
use winit::window::WindowId;

use alacritty_terminal::event::{Event as TerminalEvent, Notify, OnResize};
use alacritty_terminal::grid::{Dimensions, Scroll};
use alacritty_terminal::index::Direction;
use alacritty_terminal::term::test::TermSize;
use alacritty_terminal::term::{Term, TermMode};

use crate::cli::{ParsedOptions, WindowOptions};
#[cfg(not(windows))]
use crate::daemon::foreground_process_path;
use crate::clipboard::Clipboard;
use crate::config::UiConfig;
use crate::display::Display;
use crate::display::window::Window;
use crate::event::{ActionContext, Event, EventProxy, Mouse, SearchState, TabDragState, TouchPurpose};
#[cfg(unix)]
use crate::logging::LOG_TARGET_IPC_CONFIG;
use crate::message_bar::MessageBuffer;
use crate::scheduler::Scheduler;
use crate::tab::Tab;
use crate::{input, renderer};

/// Event context for one individual Alacritty window.
pub struct WindowContext {
    pub message_buffer: MessageBuffer,
    pub display: Display,
    pub dirty: bool,
    /// Whether the tab bar is hidden (tabs still switch via shortcuts).
    pub tab_bar_hidden: bool,
    /// All tabs for this window.
    pub tabs: Vec<Tab>,
    /// Index of the currently active tab.
    pub active_tab: usize,
    event_queue: Vec<WinitEvent<Event>>,
    cursor_blink_timed_out: bool,
    prev_bell_cmd: Option<Instant>,
    modifiers: Modifiers,
    mouse: Mouse,
    touch: TouchPurpose,
    occluded: bool,
    preserve_title: bool,
    drag_state: TabDragState,
    window_config: ParsedOptions,
    config: Rc<UiConfig>,
}

impl WindowContext {
    /// Create initial window context that does bootstrapping the graphics API we're going to use.
    pub fn initial(
        event_loop: &ActiveEventLoop,
        proxy: EventLoopProxy<Event>,
        config: Rc<UiConfig>,
        mut options: WindowOptions,
    ) -> Result<Self, Box<dyn Error>> {
        let raw_display_handle = event_loop.display_handle().unwrap().as_raw();

        let mut identity = config.window.identity.clone();
        options.window_identity.override_identity_config(&mut identity);

        // Windows has different order of GL platform initialization compared to any other platform;
        // it requires the window first.
        #[cfg(windows)]
        let window = Window::new(event_loop, &config, &identity, &mut options)?;
        #[cfg(windows)]
        let raw_window_handle = Some(window.raw_window_handle());

        #[cfg(not(windows))]
        let raw_window_handle = None;

        let gl_display = renderer::platform::create_gl_display(
            raw_display_handle,
            raw_window_handle,
            config.debug.prefer_egl,
        )?;
        let gl_config = renderer::platform::pick_gl_config(&gl_display, raw_window_handle)?;

        #[cfg(not(windows))]
        let window = Window::new(
            event_loop,
            &config,
            &identity,
            &mut options,
            #[cfg(all(feature = "x11", not(any(target_os = "macos", windows))))]
            gl_config.x11_visual(),
        )?;

        // Create context.
        let gl_context =
            renderer::platform::create_gl_context(&gl_display, &gl_config, raw_window_handle)?;

        let display = Display::new(window, gl_context, &config, false)?;

        Self::new(display, config, options, proxy)
    }

    /// Create additional context with the graphics platform other windows are using.
    pub fn additional(
        gl_config: &GlutinConfig,
        event_loop: &ActiveEventLoop,
        proxy: EventLoopProxy<Event>,
        config: Rc<UiConfig>,
        mut options: WindowOptions,
        config_overrides: ParsedOptions,
    ) -> Result<Self, Box<dyn Error>> {
        let gl_display = gl_config.display();

        let mut identity = config.window.identity.clone();
        options.window_identity.override_identity_config(&mut identity);

        // Check if new window will be opened as a tab.
        // This must be done before `Window::new()`, which unsets `window_tabbing_id`.
        #[cfg(target_os = "macos")]
        let tabbed = options.window_tabbing_id.is_some();
        #[cfg(not(target_os = "macos"))]
        let tabbed = false;

        let window = Window::new(
            event_loop,
            &config,
            &identity,
            &mut options,
            #[cfg(all(feature = "x11", not(any(target_os = "macos", windows))))]
            gl_config.x11_visual(),
        )?;

        // Create context.
        let raw_window_handle = window.raw_window_handle();
        let gl_context =
            renderer::platform::create_gl_context(&gl_display, gl_config, Some(raw_window_handle))?;

        let display = Display::new(window, gl_context, &config, tabbed)?;

        let mut window_context = Self::new(display, config, options, proxy)?;

        // Set the config overrides at startup.
        window_context.window_config = config_overrides;

        Ok(window_context)
    }

    /// Create a new window that starts with an existing tab (used for tab tear-off).
    pub fn additional_with_tab(
        gl_config: &GlutinConfig,
        event_loop: &ActiveEventLoop,
        config: Rc<UiConfig>,
        tab: crate::tab::Tab,
    ) -> Result<Self, Box<dyn Error>> {
        let gl_display = gl_config.display();

        let mut identity = config.window.identity.clone();
        let mut options = WindowOptions::default();
        options.window_identity.override_identity_config(&mut identity);

        #[cfg(target_os = "macos")]
        let tabbed = false;
        #[cfg(not(target_os = "macos"))]
        let tabbed = false;

        let window = Window::new(
            event_loop,
            &config,
            &identity,
            &mut options,
            #[cfg(all(feature = "x11", not(any(target_os = "macos", windows))))]
            gl_config.x11_visual(),
        )?;

        let raw_window_handle = window.raw_window_handle();
        let gl_context =
            renderer::platform::create_gl_context(&gl_display, gl_config, Some(raw_window_handle))?;

        let display = Display::new(window, gl_context, &config, tabbed)?;

        info!(
            "Tear-off window PTY dimensions: {:?} x {:?}",
            display.size_info.screen_lines(),
            display.size_info.columns()
        );

        let tab_title = tab.title.clone();

        let mut wc = WindowContext {
            preserve_title: false,
            display,
            tabs: vec![tab],
            active_tab: 0,
            tab_bar_hidden: false,
            drag_state: TabDragState::Idle,
            config,
            cursor_blink_timed_out: Default::default(),
            prev_bell_cmd: Default::default(),
            message_buffer: Default::default(),
            window_config: Default::default(),
            event_queue: Default::default(),
            modifiers: Default::default(),
            occluded: Default::default(),
            mouse: Default::default(),
            touch: Default::default(),
            dirty: true,
        };

        // Set initial window title.
        if wc.config.window.dynamic_title {
            wc.display.window.set_title(tab_title);
        }

        Ok(wc)
    }

    /// Create a new terminal window context.
    fn new(
        display: Display,
        config: Rc<UiConfig>,
        options: WindowOptions,
        proxy: EventLoopProxy<Event>,
    ) -> Result<Self, Box<dyn Error>> {
        let preserve_title = options.window_identity.title.is_some();

        info!(
            "PTY dimensions: {:?} x {:?}",
            display.size_info.screen_lines(),
            display.size_info.columns()
        );

        let window_id = display.window.id();

        // Create the first tab with a globally unique ID.
        let first_tab_id = NEXT_TAB_ID.fetch_add(1, Ordering::Relaxed);
        let first_tab = Tab::new(&display, &config, &options, proxy.clone(), window_id, first_tab_id)?;

        // Start cursor blinking, in case `Focused` isn't sent on startup.
        if config.cursor.style().blinking {
            let event_proxy = EventProxy::new_with_tab(proxy, window_id, first_tab.id);
            event_proxy.send_event(TerminalEvent::CursorBlinkingChange.into());
        }

        Ok(WindowContext {
            preserve_title,
            display,
            tabs: vec![first_tab],
            active_tab: 0,
            tab_bar_hidden: false,
            drag_state: TabDragState::Idle,
            config,
            cursor_blink_timed_out: Default::default(),
            prev_bell_cmd: Default::default(),
            message_buffer: Default::default(),
            window_config: Default::default(),
            event_queue: Default::default(),
            modifiers: Default::default(),
            occluded: Default::default(),
            mouse: Default::default(),
            touch: Default::default(),
            dirty: Default::default(),
        })
    }

    /// Get a reference to the active tab.
    pub fn active_tab(&self) -> &Tab {
        &self.tabs[self.active_tab]
    }

    /// Get a mutable reference to the active tab.
    #[allow(dead_code)]
    pub fn active_tab_mut(&mut self) -> &mut Tab {
        &mut self.tabs[self.active_tab]
    }

    /// Get the stable ID of the active tab.
    pub fn active_tab_id(&self) -> usize {
        self.tabs[self.active_tab].id
    }

    /// Find a tab by its stable ID.
    fn tab_index_by_id(&self, tab_id: usize) -> Option<usize> {
        self.tabs.iter().position(|t| t.id == tab_id)
    }

    /// Update the stored title of a tab and, if it's the active tab, update the window title.
    pub fn update_tab_title(&mut self, tab_id: usize, title: String) {
        if let Some(idx) = self.tab_index_by_id(tab_id) {
            self.tabs[idx].title = title.clone();
        }
        if self.active_tab_id() == tab_id
            && !self.preserve_title
            && self.config.window.dynamic_title
        {
            self.display.window.set_title(title);
        }
    }

    /// Reset a tab's title to its default and update the window title if it's active.
    pub fn reset_tab_title(&mut self, tab_id: usize) {
        let default_title = if let Some(idx) = self.tab_index_by_id(tab_id) {
            format!("Terminal {}", idx + 1)
        } else {
            return;
        };
        if let Some(idx) = self.tab_index_by_id(tab_id) {
            self.tabs[idx].title = default_title;
        }
        if self.active_tab_id() == tab_id
            && !self.preserve_title
            && self.config.window.dynamic_title
        {
            self.display.window.set_title(self.config.window.identity.title.clone());
        }
    }

    /// Create a new tab in this window. Returns the new tab's stable ID.
    pub fn create_tab(&mut self, proxy: EventLoopProxy<Event>) -> usize {
        let tab_id = NEXT_TAB_ID.fetch_add(1, Ordering::Relaxed);
        let window_id = self.display.window.id();
        let mut options = WindowOptions::default();
        #[cfg(not(windows))]
        {
            let active = self.active_tab();
            if let Ok(cwd) = foreground_process_path(active.master_fd, active.shell_pid) {
                options.terminal_options.working_directory = Some(cwd);
            }
        }
        match Tab::new(&self.display, &self.config, &options, proxy, window_id, tab_id) {
            Ok(tab) => {
                self.tabs.push(tab);
                let new_idx = self.tabs.len() - 1;
                self.select_tab(new_idx);
            },
            Err(err) => error!("Could not create tab: {err:?}"),
        }
        tab_id
    }

    /// Close the tab with the given stable ID.
    pub fn close_tab_by_id(&mut self, tab_id: usize) {
        let Some(idx) = self.tab_index_by_id(tab_id) else { return };

        self.tabs.remove(idx);

        if self.tabs.is_empty() {
            return;
        }

        // Keep active_tab pointing at the same logical tab after the removal.
        // If the closed tab was before the active one, every subsequent tab shifted
        // left by one, so we must decrement to compensate.  Then clamp in case the
        // active tab itself was the one that closed (and it was the last tab).
        if self.active_tab > idx {
            self.active_tab -= 1;
        }
        if self.active_tab >= self.tabs.len() {
            self.active_tab = self.tabs.len() - 1;
        }

        // Mark display dirty and update window title.
        self.display.damage_tracker.frame().mark_fully_damaged();
        self.dirty = true;
        self.display.pending_update.dirty = true;

        let title = self.active_tab().title.clone();
        if !self.preserve_title && self.config.window.dynamic_title {
            self.display.window.set_title(title);
        }
    }

    /// Select the tab at the given index.
    pub fn select_tab(&mut self, index: usize) {
        if index >= self.tabs.len() {
            return;
        }
        let old_idx = self.active_tab;

        // Read focus state and clear it on the old tab in one lock.
        let (was_focused, old_focus_events) = {
            let mut t = self.tabs[old_idx].terminal.lock();
            let focused = t.is_focused;
            t.is_focused = false;
            let events = t.mode().contains(TermMode::FOCUS_IN_OUT);
            (focused, events)
        };

        // Notify the old tab that it lost focus, so programs like neovim can
        // update their state (e.g. redraw unfocused cursor, stop blinking).
        if was_focused && old_focus_events {
            self.tabs[old_idx].notifier.notify(b"\x1b[O".as_slice());
        }

        self.active_tab = index;

        // Set focus state on new tab and check whether it wants focus events.
        let new_focus_events = {
            let mut t = self.tabs[index].terminal.lock();
            t.is_focused = was_focused;
            t.mode().contains(TermMode::FOCUS_IN_OUT)
        };

        // Notify the new tab that it gained focus.
        if was_focused && new_focus_events {
            self.tabs[index].notifier.notify(b"\x1b[I".as_slice());
        }

        // Force a full redraw for the new tab's content.
        self.display.damage_tracker.frame().mark_fully_damaged();
        self.dirty = true;
        self.display.pending_update.dirty = true;

        let title = self.active_tab().title.clone();
        if !self.preserve_title && self.config.window.dynamic_title {
            self.display.window.set_title(title);
        }
    }

    /// Reorder a tab from `from` index to `to` index.
    pub fn reorder_tab(&mut self, from: usize, to: usize) {
        if from == to || from >= self.tabs.len() || to >= self.tabs.len() {
            return;
        }
        let tab = self.tabs.remove(from);
        self.tabs.insert(to, tab);
        // Adjust active_tab index.
        self.active_tab = if self.active_tab == from {
            to
        } else if from < self.active_tab && to >= self.active_tab {
            self.active_tab - 1
        } else if from > self.active_tab && to <= self.active_tab {
            self.active_tab + 1
        } else {
            self.active_tab
        };
        self.dirty = true;
        self.display.pending_update.dirty = true;
    }

    /// Update the terminal window to the latest config.
    pub fn update_config(&mut self, new_config: Rc<UiConfig>) {
        let old_config = mem::replace(&mut self.config, new_config);

        // Apply ipc config if there are overrides.
        self.config = self.window_config.override_config_rc(self.config.clone());

        self.display.update_config(&self.config);

        // Update all tab terminals.
        for tab in &self.tabs {
            tab.terminal.lock().set_options(self.config.term_options());
        }

        // Reload cursor if its thickness has changed.
        if (old_config.cursor.thickness() - self.config.cursor.thickness()).abs() > f32::EPSILON {
            self.display.pending_update.set_cursor_dirty();
        }

        if old_config.font != self.config.font {
            let scale_factor = self.display.window.scale_factor as f32;
            // Do not update font size if it has been changed at runtime.
            if self.display.font_size == old_config.font.size().scale(scale_factor) {
                self.display.font_size = self.config.font.size().scale(scale_factor);
            }

            let font = self.config.font.clone().with_size(self.display.font_size);
            self.display.pending_update.set_font(font);
        }

        // Always reload the theme to account for auto-theme switching.
        self.display.window.set_theme(self.config.window.theme());

        // Update display if either padding options or resize increments were changed.
        let window_config = &old_config.window;
        if window_config.padding(1.) != self.config.window.padding(1.)
            || window_config.dynamic_padding != self.config.window.dynamic_padding
            || window_config.resize_increments != self.config.window.resize_increments
        {
            self.display.pending_update.dirty = true;
        }

        // Update title on config reload.
        if !self.preserve_title
            && (!self.config.window.dynamic_title
                || self.display.window.title() == old_config.window.identity.title)
        {
            self.display.window.set_title(self.config.window.identity.title.clone());
        }

        let opaque = self.config.window_opacity() >= 1.;

        #[cfg(target_os = "macos")]
        self.display.window.set_has_shadow(opaque);

        #[cfg(target_os = "macos")]
        self.display.window.set_option_as_alt(self.config.window.option_as_alt());

        self.display.window.set_transparent(!opaque);
        self.display.window.set_blur(self.config.window.blur);

        self.display.hint_state.update_alphabet(self.config.hints.alphabet());

        // Update cursor blinking.
        let event = Event::new(TerminalEvent::CursorBlinkingChange.into(), None);
        self.event_queue.push(event.into());

        self.dirty = true;
    }

    /// Get reference to the window's configuration.
    #[cfg(unix)]
    pub fn config(&self) -> &UiConfig {
        &self.config
    }

    /// Clear the window config overrides.
    #[cfg(unix)]
    pub fn reset_window_config(&mut self, config: Rc<UiConfig>) {
        self.message_buffer.remove_target(LOG_TARGET_IPC_CONFIG);
        self.window_config.clear();
        self.update_config(config);
    }

    /// Add new window config overrides.
    #[cfg(unix)]
    pub fn add_window_config(&mut self, config: Rc<UiConfig>, options: &ParsedOptions) {
        self.message_buffer.remove_target(LOG_TARGET_IPC_CONFIG);
        self.window_config.extend_from_slice(options);
        self.update_config(config);
    }

    /// Draw the window.
    pub fn draw(&mut self, scheduler: &mut Scheduler) {
        self.display.window.requested_redraw = false;

        if self.occluded {
            return;
        }

        self.dirty = false;

        // Force the display to process any pending display update.
        self.display.process_renderer_update();

        // Request immediate re-draw if visual bell animation is not finished yet.
        if !self.display.visual_bell.completed() {
            if self.display.window.has_frame {
                self.display.window.request_redraw();
            } else {
                self.dirty = true;
            }
        }

        // Collect per-tab display info before taking mutable borrows.
        let active_idx = self.active_tab;
        let tab_bar_hidden = self.tab_bar_hidden;
        let tab_titles: Vec<String> = self.tabs.iter().map(|t| t.title.clone()).collect();
        let drag_visual = match &self.drag_state {
            TabDragState::Dragging { tab_index, current_x, .. } => Some((*tab_index, *current_x)),
            _ => None,
        };

        // Lock the terminal through a local Arc clone to avoid borrow conflicts.
        let terminal_arc: Arc<alacritty_terminal::sync::FairMutex<alacritty_terminal::term::Term<crate::event::EventProxy>>> = Arc::clone(&self.tabs[active_idx].terminal);
        let mut terminal = terminal_arc.lock();

        // Ensure the terminal matches the current display size before drawing.
        // After a tab switch submit_display_update runs with the previous tab's terminal,
        // so the newly-active terminal may still have stale dimensions; drawing it without
        // resizing would make terminal.damage() yield line indices ≥ the damage tracker's
        // len and panic in FrameDamage::damage_line.
        {
            let display_size = self.display.size_info;
            if terminal.screen_lines() != display_size.screen_lines()
                || terminal.columns() != display_size.columns()
            {
                self.tabs[active_idx].notifier.on_resize(display_size.into());
                terminal.resize(display_size);
                self.display.damage_tracker.resize(
                    display_size.screen_lines(),
                    display_size.columns(),
                );
            }
        }

        self.display.draw(
            terminal,
            scheduler,
            &self.message_buffer,
            &self.config,
            &mut self.tabs[active_idx].search_state,
            &tab_titles,
            active_idx,
            tab_bar_hidden,
            drag_visual,
        );
    }

    /// Process events for this terminal window.
    pub fn handle_event(
        &mut self,
        #[cfg(target_os = "macos")] event_loop: &ActiveEventLoop,
        event_proxy: &EventLoopProxy<Event>,
        clipboard: &mut Clipboard,
        scheduler: &mut Scheduler,
        event: WinitEvent<Event>,
    ) {
        match event {
            WinitEvent::AboutToWait
            | WinitEvent::WindowEvent { event: WindowEvent::RedrawRequested, .. } => {
                // Skip further event handling with no staged updates.
                if self.event_queue.is_empty() {
                    return;
                }
            },
            event => {
                self.event_queue.push(event);
                return;
            },
        }

        // Pre-extract snapshot values to avoid borrow issues.
        let active_idx = self.active_tab;
        let tabs_len = self.tabs.len();
        let active_tab_id = self.tabs[active_idx].id;
        #[cfg(not(windows))]
        let master_fd = self.tabs[active_idx].master_fd;
        #[cfg(not(windows))]
        let shell_pid = self.tabs[active_idx].shell_pid;
        let old_is_searching = self.tabs[active_idx].search_state.history_index.is_some();

        // Lock through local Arc clone so the guard doesn't borrow self.
        let terminal_arc: Arc<alacritty_terminal::sync::FairMutex<alacritty_terminal::term::Term<crate::event::EventProxy>>> = Arc::clone(&self.tabs[active_idx].terminal);
        let mut terminal = terminal_arc.lock();

        // Ensure the terminal's grid dimensions match the display before processing
        // any input events. This can be stale when a tab is moved from a different-
        // sized window (the terminal keeps the old window's dimensions until a resize
        // event fires, but cursor_state() will index the grid using the new size_info
        // and panic with an out-of-bounds assertion).
        {
            let current_size = self.display.size_info;
            if terminal.screen_lines() != current_size.screen_lines()
                || terminal.columns() != current_size.columns()
            {
                self.tabs[active_idx].notifier.on_resize(current_size.into());
                terminal.resize(current_size);
            }
        }

        // Snapshot tab titles before the mutable borrow of active_tab begins.
        // Used by tab_bar_hit_test and release_tab_drag for variable-width layout.
        let tab_titles: Vec<String> = self.tabs.iter().map(|t| t.title.clone()).collect();

        {
            // Single mutable borrow of the active tab; split into sub-field borrows.
            let active_tab = &mut self.tabs[active_idx];

            let context = ActionContext {
                cursor_blink_timed_out: &mut self.cursor_blink_timed_out,
                prev_bell_cmd: &mut self.prev_bell_cmd,
                message_buffer: &mut self.message_buffer,
                inline_search_state: &mut active_tab.inline_search_state,
                search_state: &mut active_tab.search_state,
                modifiers: &mut self.modifiers,
                notifier: &mut active_tab.notifier,
                display: &mut self.display,
                mouse: &mut self.mouse,
                touch: &mut self.touch,
                dirty: &mut self.dirty,
                occluded: &mut self.occluded,
                terminal: &mut terminal,
                #[cfg(not(windows))]
                master_fd,
                #[cfg(not(windows))]
                shell_pid,
                preserve_title: self.preserve_title,
                config: &self.config,
                event_proxy,
                #[cfg(target_os = "macos")]
                event_loop,
                clipboard,
                scheduler,
                tabs_len,
                active_tab_index: active_idx,
                active_tab_id,
                drag_state: &mut self.drag_state,
                tab_bar_hidden: self.tab_bar_hidden,
                tab_titles: &tab_titles,
            };
            let mut processor = input::Processor::new(context);

            for event in self.event_queue.drain(..) {
                processor.handle_event(event);
            }
        } // active_tab mutable borrow released here.

        // Process DisplayUpdate events.
        if self.display.pending_update.dirty {
            let active_tab = &mut self.tabs[active_idx];
            Self::submit_display_update(
                &mut terminal,
                &mut self.display,
                &mut active_tab.notifier,
                &self.message_buffer,
                &mut active_tab.search_state,
                old_is_searching,
                &self.config,
                tabs_len,
                self.tab_bar_hidden,
            );
            self.dirty = true;
        }

        if self.dirty || self.mouse.hint_highlight_dirty {
            self.dirty |= self.display.update_highlighted_hints(
                &terminal,
                &self.config,
                &self.mouse,
                self.modifiers.state(),
            );
            self.mouse.hint_highlight_dirty = false;
        }

        if self.dirty
            && self.display.window.has_frame
            && !self.occluded
            && !matches!(event, WinitEvent::WindowEvent { event: WindowEvent::RedrawRequested, .. })
        {
            self.display.window.request_redraw();
        }
    }

    /// ID of this terminal context.
    pub fn id(&self) -> WindowId {
        self.display.window.id()
    }

    /// Write the ref test results to the disk.
    pub fn write_ref_test_results(&self) {
        let mut grid = self.tabs[self.active_tab].terminal.lock().grid().clone();
        grid.initialize_all();
        grid.truncate();

        let serialized_grid = json::to_string(&grid).expect("serialize grid");

        let size_info = &self.display.size_info;
        let size = TermSize::new(size_info.columns(), size_info.screen_lines());
        let serialized_size = json::to_string(&size).expect("serialize size");

        let serialized_config = format!("{{\"history_size\":{}}}", grid.history_size());

        File::create("./grid.json")
            .and_then(|mut f| f.write_all(serialized_grid.as_bytes()))
            .expect("write grid.json");

        File::create("./size.json")
            .and_then(|mut f| f.write_all(serialized_size.as_bytes()))
            .expect("write size.json");

        File::create("./config.json")
            .and_then(|mut f| f.write_all(serialized_config.as_bytes()))
            .expect("write config.json");
    }

    /// Submit the pending changes to the `Display`.
    #[allow(clippy::too_many_arguments)]
    fn submit_display_update(
        terminal: &mut Term<EventProxy>,
        display: &mut Display,
        notifier: &mut alacritty_terminal::event_loop::Notifier,
        message_buffer: &MessageBuffer,
        search_state: &mut SearchState,
        old_is_searching: bool,
        config: &UiConfig,
        tab_count: usize,
        tab_bar_hidden: bool,
    ) {
        // Compute cursor positions before resize.
        let num_lines = terminal.screen_lines();
        let cursor_at_bottom = terminal.grid().cursor.point.line + 1 == num_lines;
        let origin_at_bottom = if terminal.mode().contains(TermMode::VI) {
            terminal.vi_mode_cursor.point.line == num_lines - 1
        } else {
            search_state.direction == Direction::Left
        };

        display.handle_update(
            terminal,
            notifier,
            message_buffer,
            search_state,
            config,
            tab_count,
            tab_bar_hidden,
        );

        let new_is_searching = search_state.history_index.is_some();
        if !old_is_searching && new_is_searching {
            let display_offset = terminal.grid().display_offset();
            if display_offset == 0 && cursor_at_bottom && !origin_at_bottom {
                terminal.scroll_display(Scroll::Delta(1));
            } else if display_offset != 0 && origin_at_bottom {
                terminal.scroll_display(Scroll::Delta(-1));
            }
        }
    }
}
