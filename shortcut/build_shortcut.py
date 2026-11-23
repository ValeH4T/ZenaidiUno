"""Build the iOS Shortcut "Shadowing" (share sheet: YouTube video -> lesson on the GitHub page).

The Shortcut makes the one request YouTube refuses from the server (with the phone's own connection):
  1. GET  {space}/api/req?u=<shared link>        -> the request body for YouTube
  2. POST https://www.youtube.com/youtubei/v1/player  (body = 1)
  3. POST {space}/api/ingest?to=gh  (body = 2)   -> the page to open
  4. Open URLs (3)
"""
import plistlib
import sys
import uuid

SPACE = "https://zenaidiuno-shadowing-v2.hf.space"
OBJ = "￼"  # where a variable sits inside a text field


def output_of(uid, name="Contents of URL"):
    return {
        "Value": {"OutputName": name, "OutputUUID": uid, "Type": "ActionOutput"},
        "WFSerializationType": "WFTextTokenAttachment",
    }


def text_with_input(prefix):
    s = prefix + OBJ
    pos = len(s.encode("utf-16-le")) // 2 - 1
    return {
        "Value": {"string": s, "attachmentsByRange": {f"{{{pos}, 1}}": {"Type": "ExtensionInput"}}},
        "WFSerializationType": "WFTextTokenString",
    }


u1, u2, u3 = (str(uuid.uuid4()).upper() for _ in range(3))
actions = [
    {"WFWorkflowActionIdentifier": "is.workflow.actions.downloadurl",
     "WFWorkflowActionParameters": {"UUID": u1, "WFURL": text_with_input(f"{SPACE}/api/req?u=")}},
    {"WFWorkflowActionIdentifier": "is.workflow.actions.downloadurl",
     "WFWorkflowActionParameters": {"UUID": u2, "WFURL": "https://www.youtube.com/youtubei/v1/player",
                                    "WFHTTPMethod": "POST", "WFHTTPBodyType": "File", "ShowHeaders": False,
                                    "WFRequestVariable": output_of(u1)}},
    {"WFWorkflowActionIdentifier": "is.workflow.actions.downloadurl",
     "WFWorkflowActionParameters": {"UUID": u3, "WFURL": f"{SPACE}/api/ingest?to=gh",
                                    "WFHTTPMethod": "POST", "WFHTTPBodyType": "File", "ShowHeaders": False,
                                    "WFRequestVariable": output_of(u2)}},
    {"WFWorkflowActionIdentifier": "is.workflow.actions.openurl",
     "WFWorkflowActionParameters": {"WFInput": output_of(u3)}},
]
workflow = {
    "WFWorkflowClientVersion": "2607.0.2",
    "WFWorkflowMinimumClientVersion": 900,
    "WFWorkflowMinimumClientVersionString": "900",
    "WFWorkflowIcon": {"WFWorkflowIconStartColor": 463140863, "WFWorkflowIconGlyphNumber": 61440},
    "WFWorkflowImportQuestions": [],
    "WFWorkflowTypes": ["ActionExtension"],  # show in the share sheet
    "WFWorkflowInputContentItemClasses": ["WFURLContentItem", "WFStringContentItem", "WFSafariWebPageContentItem"],
    "WFWorkflowOutputContentItemClasses": [],
    "WFWorkflowHasShortcutInputVariables": True,
    "WFWorkflowActions": actions,
}
out = sys.argv[1] if len(sys.argv) > 1 else "Shadowing.shortcut"
with open(out, "wb") as f:
    plistlib.dump(workflow, f, fmt=plistlib.FMT_BINARY)
print("wrote", out)
