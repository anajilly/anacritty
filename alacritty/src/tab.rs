//! Tab state management.

use std::error::Error;
#[cfg(not(windows))]
use std::os::unix::io::{AsRawFd, RawFd};
use std::sync::Arc;

use winit::event_loop::EventLoopProxy;
use winit::window::WindowId;

use alacritty_terminal::event_loop::{Msg, Notifier};
use alacritty_terminal::sync::FairMutex;
use alacritty_terminal::term::Term;
use alacritty_terminal::tty;

use crate::cli::WindowOptions;
use crate::config::UiConfig;
use crate::display::Display;
use crate::event::{Event, EventProxy, InlineSearchState, SearchState};

/// A single terminal tab within a window.
pub struct Tab {
    /// Stable unique identifier for this tab.
    pub id: usize,
    /// Terminal state.
    pub terminal: Arc<FairMutex<Term<EventProxy>>>,
    /// Channel to the PTY I/O loop.
    pub notifier: Notifier,
    /// Display title for this tab.
    pub title: String,
    /// Per-tab search state.
    pub search_state: SearchState,
    /// Per-tab inline search state.
    pub inline_search_state: InlineSearchState,
    #[cfg(not(windows))]
    pub master_fd: RawFd,
    #[cfg(not(windows))]
    pub shell_pid: u32,
}

impl Tab {
    /// Create a new tab, spawning a PTY and terminal.
    pub fn new(
        display: &Display,
        config: &UiConfig,
        options: &WindowOptions,
        proxy: EventLoopProxy<Event>,
        window_id: WindowId,
        id: usize,
    ) -> Result<Self, Box<dyn Error>> {
        use alacritty_terminal::event_loop::EventLoop as PtyEventLoop;

        let mut pty_config = config.pty_config();
        options.terminal_options.override_pty_config(&mut pty_config);

        let event_proxy = EventProxy::new_with_tab(proxy, window_id, id);

        let terminal = Term::new(config.term_options(), &display.size_info, event_proxy.clone());
        let terminal = Arc::new(FairMutex::new(terminal));

        let pty = tty::new(&pty_config, display.size_info.into(), window_id.into())?;

        #[cfg(not(windows))]
        let master_fd = pty.file().as_raw_fd();
        #[cfg(not(windows))]
        let shell_pid = pty.child().id();

        let event_loop = PtyEventLoop::new(
            Arc::clone(&terminal),
            event_proxy,
            pty,
            pty_config.drain_on_exit,
            config.debug.ref_test,
        )?;

        let loop_tx = event_loop.channel();
        let _io_thread = event_loop.spawn();

        let title = format!("Terminal {}", id + 1);

        Ok(Tab {
            id,
            terminal,
            notifier: Notifier(loop_tx),
            title,
            search_state: SearchState::default(),
            inline_search_state: InlineSearchState::default(),
            #[cfg(not(windows))]
            master_fd,
            #[cfg(not(windows))]
            shell_pid,
        })
    }
}

impl Drop for Tab {
    fn drop(&mut self) {
        let _ = self.notifier.0.send(Msg::Shutdown);
    }
}
