"""
benchmark.py

The core research artifact of this project: a hand-written benchmark of
email pairs designed specifically to be invisible to word-based detection.

Every email in here is plausible, professionally written, and contains NONE
of the vocabulary a TF-IDF/keyword model learns to flag (no "urgent!!!",
no "click here", no "verify your account", no misspellings). The only
signal separating attack from legitimate, in each pair, is CONTEXT:
does this request fit this sender's established relationship and history?

Each pair shares the same scenario/category so they're a fair comparison --
the attack and its legitimate counterpart are matched, not just two
unrelated examples.

This file is deliberately data-only (no detection logic) so it can be
reused, unchanged, across every evaluation stage: content-only baseline,
+relationship layer, +LLM layer, +fusion. The benchmark has to stay fixed
for the ablation comparison across those stages to mean anything.

Each entry:
    id                 unique identifier
    category           scenario type, used to keep pairs matched
    label              1 = attack, 0 = legitimate
    sender / recipient email addresses
    prior_history      list of (request_type, days_ago) tuples describing
                        what relationship_memory.py should already know
                        about this sender/recipient pair BEFORE this email
                        arrives -- this is what makes the benchmark testable
                        against the relationship-memory layer specifically
    request_type       payment / credential / urgent / normal
    subject / body      the actual email text -- deliberately clean-worded
    why                 (not shown to any model) -- the actual reasoning
                        for why this is/isn't an attack, for scoring and
                        write-up purposes
"""

BENCHMARK = [
    # ------------------------------------------------------------------
    # Category: Vendor payment / bank detail change
    # ------------------------------------------------------------------
    {
        "id": "vendor_payment_attack",
        "category": "vendor_payment_change",
        "label": 1,
        "sender": "billing@nordholm-supplies.com",
        "recipient": "ap@yourcompany.com",
        "prior_history": [],  # first contact
        "request_type": "payment",
        "subject": "Updated remittance details for upcoming invoice",
        "body": (
            "Hi, hope you're doing well. We've recently switched banking providers "
            "on our end, so going forward please use the new account details below "
            "for this month's invoice and any future payments. Let me know if you "
            "need anything else to process it on your side. Thanks for your continued "
            "partnership."
        ),
        "why": "First-ever contact from this address, immediately asking for a "
               "banking change -- classic pretexting. No suspicious wording at all.",
    },
    {
        "id": "vendor_payment_legit",
        "category": "vendor_payment_change",
        "label": 0,
        "sender": "billing@nordholm-supplies.com",
        "recipient": "ap@yourcompany.com",
        "prior_history": [("payment", 400), ("payment", 340), ("payment", 280),
                           ("payment", 220), ("payment", 160), ("payment", 100)],
        "request_type": "payment",
        "subject": "Updated remittance details for upcoming invoice",
        "body": (
            "Hi, hope you're doing well. We've recently switched banking providers "
            "on our end, so going forward please use the new account details below "
            "for this month's invoice and any future payments. Let me know if you "
            "need anything else to process it on your side. Thanks for your continued "
            "partnership."
        ),
        "why": "Identical wording to the attack -- but this vendor has been "
               "invoicing this company routinely for over a year.",
    },

    # ------------------------------------------------------------------
    # Category: Executive gift card request
    # ------------------------------------------------------------------
    {
        "id": "gift_card_attack",
        "category": "executive_gift_card",
        "label": 1,
        "sender": "j.harmon.ceo@yourcompany-exec.com",
        "recipient": "assistant@yourcompany.com",
        "prior_history": [],
        "request_type": "urgent",
        "subject": "Quick favor before my flight",
        "body": (
            "Hey, I'm about to board and need to sort out a client appreciation "
            "gift before I land. Could you pick up a few gift cards and send me "
            "the codes? I'll expense it properly when I'm back online. Really "
            "appreciate you handling this while I'm traveling."
        ),
        "why": "Domain is a lookalike (yourcompany-exec.com, not yourcompany.com), "
               "no prior contact, urgency framed around unavailability -- a "
               "classic real-world BEC pattern with zero suspicious vocabulary.",
    },
    {
        "id": "gift_card_legit",
        "category": "executive_gift_card",
        "label": 0,
        "sender": "j.harmon@yourcompany.com",
        "recipient": "assistant@yourcompany.com",
        "prior_history": [("normal", 200), ("normal", 150), ("normal", 90),
                           ("urgent", 60), ("normal", 30)],
        "request_type": "urgent",
        "subject": "Quick favor before my flight",
        "body": (
            "Hey, I'm about to board and need to sort out a client appreciation "
            "gift before I land. Could you pick up a few gift cards and send me "
            "the codes? I'll expense it properly when I'm back online. Really "
            "appreciate you handling this while I'm traveling."
        ),
        "why": "Correct company domain, and this exec has an established "
               "pattern of similar last-minute asks with this assistant.",
    },

    # ------------------------------------------------------------------
    # Category: IT credential / access request
    # ------------------------------------------------------------------
    {
        "id": "credential_attack",
        "category": "credential_request",
        "label": 1,
        "sender": "helpdesk@yourcompany-support.net",
        "recipient": "employee@yourcompany.com",
        "prior_history": [],
        "request_type": "credential",
        "subject": "Migrating your account to the new SSO provider",
        "body": (
            "As part of this quarter's infrastructure update, we're migrating "
            "accounts to our new single sign-on provider. To complete your "
            "migration, please confirm your current username and temporary "
            "access code at your earliest convenience so we can finish "
            "provisioning your new profile before the cutover this weekend."
        ),
        "why": "Lookalike domain, first contact, credential request framed as "
               "routine IT process -- no urgency language or misspellings a "
               "keyword filter would catch.",
    },
    {
        "id": "credential_legit",
        "category": "credential_request",
        "label": 0,
        "sender": "helpdesk@yourcompany.com",
        "recipient": "employee@yourcompany.com",
        "prior_history": [("normal", 500), ("credential", 300), ("normal", 200),
                           ("credential", 100), ("normal", 45)],
        "request_type": "credential",
        "subject": "Migrating your account to the new SSO provider",
        "body": (
            "As part of this quarter's infrastructure update, we're migrating "
            "accounts to our new single sign-on provider. To complete your "
            "migration, please confirm your current username and temporary "
            "access code at your earliest convenience so we can finish "
            "provisioning your new profile before the cutover this weekend."
        ),
        "why": "Correct internal domain, and this helpdesk address has "
               "legitimately requested credential-related actions twice before.",
    },

    # ------------------------------------------------------------------
    # Category: HR / payroll detail change
    # ------------------------------------------------------------------
    {
        "id": "payroll_attack",
        "category": "payroll_change",
        "label": 1,
        "sender": "m.delgado@yourcompany.co",  # .co not .com
        "recipient": "payroll@yourcompany.com",
        "prior_history": [],
        "request_type": "payment",
        "subject": "Direct deposit update",
        "body": (
            "Hi, I recently switched banks and need to update my direct deposit "
            "information before the next pay cycle. Could you update my account "
            "details on file? I've attached the new routing and account number. "
            "Thanks for taking care of this."
        ),
        "why": "Domain typo-squat (.co instead of .com), no prior contact "
               "history at all with payroll -- a real employee would already "
               "have an established record.",
    },
    {
        "id": "payroll_legit",
        "category": "payroll_change",
        "label": 0,
        "sender": "m.delgado@yourcompany.com",
        "recipient": "payroll@yourcompany.com",
        "prior_history": [("normal", 700), ("normal", 500), ("payment", 20)],
        "request_type": "payment",
        "subject": "Direct deposit update",
        "body": (
            "Hi, I recently switched banks and need to update my direct deposit "
            "information before the next pay cycle. Could you update my account "
            "details on file? I've attached the new routing and account number. "
            "Thanks for taking care of this."
        ),
        "why": "Correct domain, existing employee record, and even made a "
               "similar request recently -- fits an established pattern.",
    },

    # ------------------------------------------------------------------
    # Category: New vendor first invoice
    # ------------------------------------------------------------------
    {
        "id": "new_vendor_attack",
        "category": "new_vendor_invoice",
        "label": 1,
        "sender": "accounts@brightpath-consult.com",
        "recipient": "ap@yourcompany.com",
        "prior_history": [],
        "request_type": "payment",
        "subject": "Invoice #1042 for consulting services rendered",
        "body": (
            "Please find attached invoice #1042 for the consulting engagement "
            "completed last month. Payment can be made via wire transfer to the "
            "account details included in the invoice. Let us know if you have "
            "any questions about the scope covered."
        ),
        "why": "No engagement was ever contracted with this vendor -- purely "
               "fabricated context, professionally worded, first contact.",
    },
    {
        "id": "new_vendor_legit",
        "category": "new_vendor_invoice",
        "label": 0,
        "sender": "accounts@brightpath-consult.com",
        "recipient": "ap@yourcompany.com",
        "prior_history": [("normal", 45), ("normal", 20)],  # onboarding emails
        "request_type": "payment",
        "subject": "Invoice #1042 for consulting services rendered",
        "body": (
            "Please find attached invoice #1042 for the consulting engagement "
            "completed last month. Payment can be made via wire transfer to the "
            "account details included in the invoice. Let us know if you have "
            "any questions about the scope covered."
        ),
        "why": "This vendor was onboarded a month ago with normal "
               "correspondence before this, matching a real engagement.",
    },

    # ------------------------------------------------------------------
    # Category: Colleague urgent document request
    # ------------------------------------------------------------------
    {
        "id": "colleague_urgent_attack",
        "category": "colleague_document_request",
        "label": 1,
        "sender": "r.chen@yourcompany.com",
        "recipient": "finance@yourcompany.com",
        "prior_history": [],
        "request_type": "urgent",
        "subject": "Need the Q3 client list before my call",
        "body": (
            "Hi, I'm covering for someone on the client relations team and need "
            "the full Q3 client contact list ahead of a call in the next hour. "
            "Could you send it over as a spreadsheet? Thanks so much, tight "
            "turnaround today."
        ),
        "why": "This 'colleague' has never contacted finance before, and the "
               "framing ('covering for someone') is a plausible-sounding "
               "excuse for why there's no history -- worth flagging.",
    },
    {
        "id": "colleague_urgent_legit",
        "category": "colleague_document_request",
        "label": 0,
        "sender": "r.chen@yourcompany.com",
        "recipient": "finance@yourcompany.com",
        "prior_history": [("normal", 100), ("urgent", 60), ("normal", 30), ("urgent", 10)],
        "request_type": "urgent",
        "subject": "Need the Q3 client list before my call",
        "body": (
            "Hi, I'm covering for someone on the client relations team and need "
            "the full Q3 client contact list ahead of a call in the next hour. "
            "Could you send it over as a spreadsheet? Thanks so much, tight "
            "turnaround today."
        ),
        "why": "Same wording, but this person has a real pattern of "
               "similar urgent requests to finance over several months.",
    },

    # ------------------------------------------------------------------
    # Category: Cloud storage / document access
    # ------------------------------------------------------------------
    {
        "id": "cloud_access_attack",
        "category": "cloud_access_share",
        "label": 1,
        "sender": "d.morris@partner-firm-legal.com",
        "recipient": "legal@yourcompany.com",
        "prior_history": [],
        "request_type": "credential",
        "subject": "Access to the shared due diligence folder",
        "body": (
            "Hi, following up on the deal discussion -- could you grant me "
            "access to the shared due diligence folder? I'll need editor "
            "permissions so I can upload our side's documents directly. "
            "Happy to hop on a call if that's easier."
        ),
        "why": "References a 'deal discussion' that never happened with "
               "this specific sender -- fabricated shared context, no prior "
               "contact at all, asking for elevated access.",
    },
    {
        "id": "cloud_access_legit",
        "category": "cloud_access_share",
        "label": 0,
        "sender": "d.morris@partner-firm-legal.com",
        "recipient": "legal@yourcompany.com",
        "prior_history": [("normal", 30), ("normal", 15), ("normal", 5)],
        "request_type": "credential",
        "subject": "Access to the shared due diligence folder",
        "body": (
            "Hi, following up on the deal discussion -- could you grant me "
            "access to the shared due diligence folder? I'll need editor "
            "permissions so I can upload our side's documents directly. "
            "Happy to hop on a call if that's easier."
        ),
        "why": "This contact has been in active, real correspondence about "
               "this exact deal for the past month.",
    },
]


def summary():
    attacks = [b for b in BENCHMARK if b["label"] == 1]
    legit = [b for b in BENCHMARK if b["label"] == 0]
    print(f"Benchmark size: {len(BENCHMARK)} emails ({len(attacks)} attacks, {len(legit)} legitimate look-alikes)")
    print(f"Categories: {len(set(b['category'] for b in BENCHMARK))}")
    print()
    for cat in sorted(set(b["category"] for b in BENCHMARK)):
        pair = [b for b in BENCHMARK if b["category"] == cat]
        print(f"  {cat}")
        for p in pair:
            tag = "ATTACK" if p["label"] == 1 else "legit "
            print(f"    [{tag}] {p['id']:28s} prior_contacts={len(p['prior_history']):>2}  \"{p['subject']}\"")


if __name__ == "__main__":
    summary()
