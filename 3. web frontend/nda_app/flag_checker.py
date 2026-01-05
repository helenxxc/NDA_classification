import re

flag_rules = {
    "confidentiality obligations": {
        "keywords": [
            "unilateral", "one-way", "one way",
            'company ("disclosing party")', 'company ("discloser")',
            'corporation ("disclosing party")', 'corporation ("discloser")',
            'llc ("disclosing party")', 'llc ("discloser")',
            'inc. ("disclosing party")', 'inc. ("discloser")',
            'incorporated ("disclosing party")', 'incorporated ("discloser")',
            'co. ("disclosing party")', 'co. ("discloser")',
            "residual", "residuals", "memory", "memories", "unaided", "perpetual"
        ],
        "sentences": [
            "was communicated by disclosing party to an unaffiliated third party free of any obligation of confidence",
            "was communicated by disclosing party to a third party without an obligation of confidence",
            "was disclosed by discloser to a third party without an obligation of confidentiality",
            "is or has been disclosed without confidentiality obligations to a third party by the discloser"
        ]
    },

    "remedies": {
        "must_have": ["injunction", "injunctive", "equitable"]
    },

    "privacy & security": {
        "keywords": ["personal data", "personal information", "pii", "gdpr", "ccpa", "cpra", "privacy"]
    },

    "indirect damages waiver": {
        "keywords": [
            "special, incidental, indirect or consequential damages",
            "punitive", "exemplary", "consequential", "indirect", "incidental"
        ]
    },

    "non-competition": {
        "keywords": ["compete", "competition", "non-compete", "non competition", "circumvent", "circumvention"]
    },

    "non-solicitation": {
        "keywords": [
            "non-solicitation", "solicit", "non-solicit", "non-servicing",
            "nonsolicitation", "nonsolicit", "non servicing",
            "no solicit", "no-solicit", "induce", "sever"
        ]
    },

    "indemnification": {
        "keywords": [
            "indemnification", "indemnity", "hold-harmless", "hold harmless",
            "indemnify", "indemnified", "defend"
        ]
    },

    "governing law": {
        "keywords": ["texas", "italy", "italian", "massachusetts", "louisiana"]
    }
}



def check_flag(category, text):
    category = str(category).strip().lower()

    text_clean = re.sub(r'[^\w\s]', ' ', str(text).lower())

    rules = flag_rules.get(category)
    if not rules:
        return False

    if "sentences" in rules:
        for s in rules["sentences"]:
            if s.lower() in text.lower():
                return True

    if "keywords" in rules:
        for kw in rules["keywords"]:
            words = kw.lower().split()
            if all(w in text_clean.split() for w in words):
                return True

    if category == "remedies":
        must_have = ["injunction", "injunctive", "equitable"]
        for kw in must_have:
            if kw.lower() in text_clean:
                return False
        return True

    if "must_have" in rules:
        for kw in rules["must_have"]:
            if kw.lower() in text_clean:
                return False
        return True

    return False