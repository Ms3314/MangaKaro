"""Prompts and JSON schemas for the vision/text model stages. Edit here to change what the model is asked."""

EMOTIONS = ["neutral", "happy", "sad", "angry", "surprised", "fearful", "disgusted",
            "embarrassed", "excited", "sarcastic", "determined", "pained", "other"]

READ_SCHEMA = {"type": "object", "properties": {"regions": {"type": "array", "items": {
    "type": "object", "properties": {"id": {"type": "string"}, "source_text": {"type": "string"}},
    "required": ["id", "source_text"]}}}, "required": ["regions"]}

EMOTION_SCHEMA = {"type": "object", "properties": {
    "scene": {"type": "string"},
    "regions": {"type": "array", "items": {"type": "object", "properties": {
        "id": {"type": "string"},
        "speaker": {"type": "string"},
        "emotion": {"type": "string", "enum": EMOTIONS},
        "intensity": {"type": "number"},
        "style": {"type": "object", "properties": {
            "weight": {"type": "string", "enum": ["light", "normal", "bold"]},
            "slant": {"type": "string", "enum": ["upright", "italic"]},
            "size": {"type": "string", "enum": ["small", "medium", "large"]},
            "shape": {"type": "string", "enum": ["normal", "jagged", "wavy", "outlined"]}},
            "required": ["weight", "slant", "size", "shape"]},
    }, "required": ["id", "speaker", "emotion", "intensity", "style"]}}}, "required": ["scene", "regions"]}

TRANSLATE_SCHEMA = {"type": "object", "properties": {"regions": {"type": "array", "items": {
    "type": "object", "properties": {"id": {"type": "string"}, "translation": {"type": "string"},
                                     "flag": {"type": ["string", "null"]}},
    "required": ["id", "translation", "flag"]}}}, "required": ["regions"]}

READ_PROMPT = """This is one manga page. Red boxes mark text regions, each with a numbered tag (like 2.1) at its top-left corner, \
outside the box: the tag is a label, not part of the text. Regions and pixel boxes [x1,y1,x2,y2]:

{regions}

Image 2 is a contact sheet with a close-up of the inside of each box, labelled with its id. Use it to read the exact characters.

Transcribe the text of EVERY listed id exactly as written, and nothing else. Japanese is usually vertical: read each column top to \
bottom, columns right to left. Ignore furigana (tiny reading hints beside kanji). Do not correct, translate or guess characters you \
cannot see. If a box has no readable text, use "". Return one entry per id, in the same order."""

EMOTION_PROMPT = """This is one manga page. Red boxes mark text regions with numbered tags. The text in each region has already been \
transcribed:

{regions}

Do NOT translate. For every id return:
- speaker: who is speaking, in a few English words (e.g. "boy with goggles", "narration", "old man"), or "sound effect"
- emotion: one of {emotions}, judged from the character's face, pose and the scene as well as the words
- intensity: 0.0 (calm) to 1.0 (extreme)
- style: how the ORIGINAL lettering looks: weight (light/normal/bold), slant (upright/italic), size (small/medium/large relative to \
the rest of the page), shape (normal/jagged for shouting/wavy for trembling or sarcasm/outlined for big sound effects)
Also give "scene": one short English sentence on what is happening on this page. Describe only what you can see; do not guess \
character or series names. Return one entry per id, in the same order."""

TRANSLATE_PROMPT = """You are translating Japanese manga dialogue into natural English for speech bubbles.

Scene: {scene}

Lines in reading order (id | speaker | emotion, intensity | Japanese):
{lines}

Translate every line. Read the page as one conversation: keep each speaker's voice and pronouns consistent, and let later lines \
clarify earlier ones. Match the emotion. Keep it short enough for a speech bubble. The Japanese was transcribed from an image and may \
contain small character errors: if a line looks garbled, translate the most plausible meaning from context. Use "" for an empty line. \
flag: null, or a short reason if the text is obscene or problematic. Return one entry per id, in the same order."""
