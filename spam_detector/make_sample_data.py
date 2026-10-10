"""Generates a small synthetic demo dataset (data/emails.csv) so the project runs
out of the box. For real accuracy, replace data/emails.csv with a real dataset
(see README)."""
import random
import pandas as pd

random.seed(42)

spam_subjects = [
    "WINNER! You have been selected for a ${amt} gift card",
    "Claim your free {prize} now",
    "URGENT: Your account will be suspended, verify now",
    "Congratulations! You won the lottery of ${amt}",
    "Cheap {drug} online, no prescription needed",
    "Make ${amt} a day working from home!!!",
    "Act now: limited time offer, 90% off {prize}",
    "Your PayPal account is locked. Click here to restore access",
    "You have an unclaimed inheritance of ${amt}",
    "Hot singles in your area want to meet you",
    "Increase your credit score instantly, guaranteed",
    "FREE {prize}!!! Reply YES to claim",
]
spam_bodies = [
    "Dear customer, click http://bit.ly/{tok} to claim your reward. Offer expires in 24 hours. Send your bank details to receive ${amt}.",
    "Congratulations!!! You are our lucky winner. Call now {phone} or visit www.{tok}-prizes.com to collect your {prize}.",
    "We detected unusual activity. Verify your password immediately at http://secure-{tok}.com/login or your account will be closed.",
    "Buy {drug} at 80% discount. No prescription. Discreet shipping. Order now, limited stock! Unsubscribe not available.",
    "Earn ${amt} per week from home. No experience needed. Just send a small registration fee to get started. Act fast!!!",
    "FINAL NOTICE: you have won ${amt}. Reply with your full name, address and card number to claim your free {prize}.",
]
ham_subjects = [
    "Meeting notes from {day}", "Lunch on {day}?", "Project {proj} status update",
    "Invoice for {proj} attached", "Re: question about the assignment",
    "Your flight itinerary for next week", "Happy birthday!", "Team standup moved to {day}",
    "Can you review my pull request?", "Dinner plans this weekend", "Reminder: dentist appointment {day}",
    "Notes from the {proj} workshop", "Family photos from the trip", "Quarterly report draft",
]
ham_bodies = [
    "Hi, just a quick note to say the {proj} meeting is moved to {day} at 3pm. Let me know if that works for you. Thanks!",
    "Hey, are you free for lunch on {day}? There's a new place near the office I wanted to try.",
    "Please find attached the latest draft of the {proj} report. Let me know your comments before the end of the week.",
    "Thanks for your help yesterday. I've pushed the changes to the repo, could you take a look when you get a chance?",
    "Hi mom, we arrived safely. The weather is great and the kids loved the beach. Will call you tonight.",
    "Reminder that the {proj} deadline is next {day}. Let's sync tomorrow morning to go over the remaining tasks.",
    "Just wanted to follow up on our conversation about the schedule. I think we can finish the {proj} work by Friday.",
]
days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]
projs = ["Apollo", "Orion", "Atlas", "Phoenix", "Nova", "budget", "onboarding"]
prizes = ["iPhone", "laptop", "vacation", "gift card", "cruise"]
drugs = ["pills", "meds", "tablets"]

def fill(t):
    return (t.format(amt=random.choice([500, 1000, 5000, 100000]),
                     prize=random.choice(prizes), drug=random.choice(drugs),
                     tok="".join(random.choices("abcdefghxyz123", k=6)),
                     phone="+1-800-%03d-%04d" % (random.randint(100, 999), random.randint(0, 9999)),
                     day=random.choice(days), proj=random.choice(projs)))

rows = []
for _ in range(400):
    rows.append(("spam", fill(random.choice(spam_subjects)) + " " + fill(random.choice(spam_bodies))))
for _ in range(400):
    rows.append(("ham", fill(random.choice(ham_subjects)) + " " + fill(random.choice(ham_bodies))))
random.shuffle(rows)
pd.DataFrame(rows, columns=["label", "text"]).to_csv("data/emails.csv", index=False)
print("Wrote data/emails.csv with", len(rows), "rows")
