# Code taken from
# https://github.com/h3llrais3r/Deluge-PreventSuspendPlus/blob/7984c79a36d7cd32dacb7f6d248061b52a9c49e7/preventsuspendplus/core.py

import os
import atexit


class DummySuspendManager:
    def __init__(self, app_name: str, reason: str):
        self.app_name = app_name
        self.reason = reason

    def inhibit(self):
        pass

    def uninhibit(self):
        pass

    def __enter__(self):
        self.inhibit()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.uninhibit()
        return self


class DBusInhibitor(DummySuspendManager):
    def __init__(self, app_name: str, reason: str, name, path, interface, method=None):
        super(DBusInhibitor, self).__init__(app_name, reason)

        if method is None:
            method = ["Inhibit", "UnInhibit"]
        self.name = name
        self.path = path
        self.interface_name = interface
        self.cookie = None

        # noinspection PyUnresolvedReferences
        import dbus
        bus = dbus.SessionBus()
        devobj = bus.get_object(self.name, self.path)
        self.iface = dbus.Interface(devobj, self.interface_name)
        # Check we have the right attributes
        self._inhibit = getattr(self.iface, method[0])
        self._uninhibit = getattr(self.iface, method[1])

    def inhibit(self):
        self.cookie = self._inhibit(self.app_name, self.reason)

    def uninhibit(self):
        self._uninhibit(self.cookie)


class GnomeSessionInhibitor(DBusInhibitor):
    TOPLEVEL_XID = 0
    INHIBIT_SUSPEND = 4

    def __init__(self, app_name: str, reason: str):
        super(GnomeSessionInhibitor, self).__init__(
            app_name,
            reason,
            "org.gnome.SessionManager",
            "/org/gnome/SessionManager",
            "org.gnome.SessionManager",
            ["Inhibit", "Uninhibit"]
        )

    def inhibit(self):
        self.cookie = self._inhibit(self.app_name,
                                    GnomeSessionInhibitor.TOPLEVEL_XID,
                                    self.reason,
                                    GnomeSessionInhibitor.INHIBIT_SUSPEND)


class WindowsInhibitor(DummySuspendManager):
    """https://msdn.microsoft.com/en-us/library/windows/desktop/aa373208(v=vs.85).aspx"""
    ES_CONTINUOUS = 0x80000000
    ES_SYSTEM_REQUIRED = 0x00000001

    def __init__(self, app_name: str, reason: str):
        super(WindowsInhibitor, self).__init__(app_name, reason)

    def inhibit(self):
        import ctypes
        ctypes.windll.kernel32.SetThreadExecutionState(
            WindowsInhibitor.ES_CONTINUOUS | WindowsInhibitor.ES_SYSTEM_REQUIRED)

    def uninhibit(self):
        import ctypes
        ctypes.windll.kernel32.SetThreadExecutionState(WindowsInhibitor.ES_CONTINUOUS)


class SuspendManager(DummySuspendManager):
    def __init__(self, app_name: str, reason: str):
        super(SuspendManager, self).__init__(app_name, reason)

        self.inhibited = False
        if os.name == "nt":
            self.os_sleep: DummySuspendManager = WindowsInhibitor(app_name, reason)
        elif os.name == "posix":
            self.os_sleep: DummySuspendManager = GnomeSessionInhibitor(app_name, reason)
        else:
            self.os_sleep: DummySuspendManager = DummySuspendManager(app_name, reason)

        atexit.register(self.__cleanup)

    def __cleanup(self):
        if self.inhibited:
            self.uninhibit()

    def inhibit(self):
        self.inhibited = True
        try:
            self.os_sleep.inhibit()
        # noinspection PyBroadException
        except Exception:
            pass

    def uninhibit(self):
        self.inhibited = False
        try:
            self.os_sleep.uninhibit()
        # noinspection PyBroadException
        except Exception:
            pass
