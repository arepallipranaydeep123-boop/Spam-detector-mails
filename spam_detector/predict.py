"""Classify text from the command line.

  python predict.py "You won a free iPhone, click here!"
  python predict.py            # interactive mode
"""
import sys
import joblib

model = joblib.load("models/spam_model.joblib")


def classify(text):
    proba = model.predict_proba([text])[0][1]
    return ("SPAM" if proba >= 0.5 else "HAM"), proba


if __name__ == "__main__":
    if len(sys.argv) > 1:
        label, p = classify(" ".join(sys.argv[1:]))
        print(f"{label} (spam probability {p:.2%})")
    else:
        print("Enter an email (blank line to quit):")
        while True:
            t = input("> ").strip()
            if not t:
                break
            label, p = classify(t)
            print(f"  {label} (spam probability {p:.2%})")
