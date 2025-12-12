import re

class Text_Cleaner:
    def __init__(self, text):
        self.text = text

    def clean_text(self):
        """
        Basic cleanup: normalize whitespace, remove repeated header/footer artifacts.
        Add stronger cleanup rules as needed.
        """
        if not self.text:
            return self.text
        # remove multiple newlines
        text = re.sub(r'\n\s*\n+', '\n\n', self.text)
        # fix weird whitespace
        text = re.sub(r'[ \t]+', ' ', text)
        # strip
        text = text.strip()
        return text