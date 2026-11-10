' APB training-machine agent: launch silently in background (no console window).
'
' Why VBS: pythonw.exe has no console, but a .bat that calls it still flashes a
' window. WScript.Shell.Run with window style 0 launches truly hidden.
'
' Usage: double-click, or:
'   wscript //nologo start_agent_hidden.vbs
'
' Stop:  stop_agent.bat  (or: python tools\agent_daemon.py --stop)
' Status: python tools\agent_daemon.py --status

Option Explicit
Dim fso, shell, root, py, cmd
Set fso = CreateObject("Scripting.FileSystemObject")
Set shell = CreateObject("WScript.Shell")

root = fso.GetParentFolderName(WScript.ScriptFullName)
shell.CurrentDirectory = root

' Prefer the bundled portable runtime, fall back to system pythonw.
If fso.FileExists(root & "\runtime\python\pythonw.exe") Then
  py = """" & root & "\runtime\python\pythonw.exe"""
Else
  py = "pythonw"
End If

cmd = py & " """ & root & "\tools\agent_daemon.py"""
shell.Run cmd, 0, False
