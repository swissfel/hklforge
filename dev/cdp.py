"""Minimal Chrome DevTools driver for dev/check_viewer.py: start headless Chrome, open a URL, evaluate JS."""
import json
import os
import subprocess
import tempfile
import time
import urllib.request

import websocket


class Chrome:
    def __init__(self, port=9333, width=1500, height=1100):
        self.port = port
        self.profile = tempfile.mkdtemp(prefix="cdp-")
        self.proc = subprocess.Popen(
            ["google-chrome", "--headless=new", f"--remote-debugging-port={port}", "--no-first-run",
             "--no-default-browser-check", f"--user-data-dir={self.profile}", "--enable-unsafe-swiftshader",
             "--use-angle=swiftshader", "--ignore-gpu-blocklist", f"--window-size={width},{height}",
             "--allow-file-access-from-files", "about:blank"],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        for _ in range(100):
            try:
                tabs = json.load(urllib.request.urlopen(f"http://127.0.0.1:{port}/json"))
                page = [t for t in tabs if t["type"] == "page"][0]
                break
            except Exception:
                time.sleep(0.1)
        self.ws = websocket.create_connection(page["webSocketDebuggerUrl"], timeout=120)
        self.i = 0
        self.errors = []
        self.call("Runtime.enable")
        self.call("Page.enable")

    def call(self, method, **params):
        self.i += 1
        my = self.i
        self.ws.send(json.dumps({"id": my, "method": method, "params": params}))
        while True:
            m = json.loads(self.ws.recv())
            if m.get("method") == "Runtime.exceptionThrown":
                d = m["params"]["exceptionDetails"]
                self.errors.append((d.get("exception", {}).get("description") or d.get("text"))[:400])
            if m.get("method") == "Runtime.consoleAPICalled" and m["params"]["type"] == "error":
                self.errors.append("console: " + " ".join(str(a.get("value", a.get("description")))
                                                            for a in m["params"]["args"])[:400])
            if m.get("id") == my:
                if "error" in m:
                    raise RuntimeError(m["error"])
                return m.get("result", {})

    def goto(self, url, wait=2.0):
        self.call("Page.navigate", url=url)
        self.sleep(wait)

    def sleep(self, s):
        end = time.time() + s
        while time.time() < end:
            self.call("Runtime.evaluate", expression="1")   # pumps events (errors)
            time.sleep(0.1)

    def js(self, expr):
        r = self.call("Runtime.evaluate", expression=expr, returnByValue=True, awaitPromise=True)
        if "exceptionDetails" in r:
            raise RuntimeError(r["exceptionDetails"].get("exception", {}).get("description", r["exceptionDetails"]))
        return r["result"].get("value")

    def click_at(self, x, y):
        for t in ("mousePressed", "mouseReleased"):
            self.call("Input.dispatchMouseEvent", type=t, x=x, y=y, button="left", clickCount=1)

    def screenshot(self, path):
        import base64
        data = self.call("Page.captureScreenshot", format="png")["data"]
        open(path, "wb").write(base64.b64decode(data))

    def close(self):
        try:
            self.ws.close()
        finally:
            self.proc.terminate()
            self.proc.wait(10)
