from typing import Dict, Union, Tuple, Set

class SurnameIndex:
    surnames = set()

    @staticmethod
    def initialize():
        if len(SurnameIndex.surnames) == 0:
            with open("resources/names/dict.csv", "r", encoding="utf-8") as file:
                for index, row in enumerate(file):
                    parts = row.split(";")
                    if index > 0:
                        SurnameIndex.surnames.add(parts[0].strip())

    @staticmethod
    def is_name(name: str) -> bool:
        return name in SurnameIndex.surnames


class FamousNameIndex:
    famous_names = set()

    @staticmethod
    def initialize():
        if len(FamousNameIndex.famous_names) == 0:
            with open("resources/names/famous_names.txt", "r", encoding="utf-8") as file:
                for row in file:
                    FamousNameIndex.famous_names.add(row.strip())

    @staticmethod
    def is_famous_name(name: str) -> bool:
        name = name.replace("Jr.", "")
        name = name.replace("Sir", "")
        name = name.replace("'", "")

        # "Hartung, Daniel" instead of "Daniel Hartung"
        if name.find(",") > 0:
            parts = name.split(",")
            if len(parts) < 2:
                return False
            name = parts[1].strip() + parts[0].strip()

        parts = name.split(" ")
        if len(parts) <= 2:
            return name in FamousNameIndex.famous_names
        else:
            # Try different amount of middle names
            for i in range(1, len(parts) - 1):
                name = " ".join(parts[0:i]) + " " + parts[-1]
                if name in FamousNameIndex.famous_names:
                    return True
            return False


class Pronouns:
    pronouns: Dict[str, Tuple[int, bool]] = None

    @staticmethod
    def initialize():
        if Pronouns.pronouns is None:
            Pronouns.pronouns = dict()
            with open("resources/linguistic/pronouns.csv", "r") as file:
                for index, row in enumerate(file):
                    if index > 0:
                        parts = row.strip().split(",")
                        Pronouns.pronouns[parts[0]] = (int(parts[1]), parts[2] == "plural")

    @staticmethod
    def is_pronoun(word: str):
        return word in Pronouns.pronouns

    @staticmethod
    def pronoun_type(word: str) -> Union[None, Tuple[int, bool]]:
        if word in Pronouns.pronouns:
            return Pronouns.pronouns[word]
        else:
            return None


class TopWordIndex:
    words: Set[str] = None

    @staticmethod
    def initialize():
        if TopWordIndex.words is None:
            TopWordIndex.words = set()
            with open("resources/linguistic/google-10000-english.txt", "r") as file:
                for index, row in enumerate(file):
                    TopWordIndex.words.add(row.strip())

    @staticmethod
    def is_top_word(word: str):
        return word in TopWordIndex.words


class ProfanityWordIndex:
    words: Set[str] = None

    @staticmethod
    def initialize():
        if ProfanityWordIndex.words is None:
            ProfanityWordIndex.words = set()
            with open("resources/linguistic/profanity-words.txt", "r") as file:
                for index, row in enumerate(file):
                    ProfanityWordIndex.words.add(row.strip())

    @staticmethod
    def is_profanity_word(word: str):
        return word in ProfanityWordIndex.words
