"""Build data/sample_docs/nimbus_data_processing_agreement.pdf.

A fictional agreement laid out like a real one: a running header and footer on every page,
numbered clauses, a table, and paragraphs that continue across page breaks. Those are the
things that make PDF question answering hard, so the sample exercises them.

    uv run python scripts/make_sample_pdf.py
"""

from pathlib import Path

from fpdf import FPDF

OUT = Path(__file__).resolve().parents[1] / "data" / "sample_docs" / (
    "nimbus_data_processing_agreement.pdf"
)

TITLE = "Nimbus Labs Ltd. - Data Processing Agreement (v2.1)"
# Start each text block at the left margin on a new line.
NEXT_LINE = {"new_x": "LMARGIN", "new_y": "NEXT"}

CLAUSES: list[tuple[str, list[str]]] = [
    ("1. Parties and Scope", [
        "This Data Processing Agreement (the \"DPA\") is entered into between Nimbus Labs Ltd., "
        "a company registered in Ireland with its registered office in Dublin (\"Nimbus\" or "
        "the \"Processor\"), and the customer identified in the order form (the \"Customer\" "
        "or the \"Controller\"). It forms part of the Nimbus Analytics Terms of Service.",
        "This DPA applies to all personal data that Nimbus processes on behalf of the Customer "
        "while providing the Nimbus Analytics platform. Version 2.1 of this DPA takes effect on "
        "1 March 2026 and replaces all earlier versions.",
    ]),
    ("2. Definitions", [
        "\"Personal Data\", \"Processing\", \"Controller\" and \"Processor\" have the meanings "
        "given in the EU General Data Protection Regulation (GDPR). \"Sub-processor\" means any "
        "third party engaged by Nimbus that processes Personal Data on behalf of the Customer. "
        "\"Security Incident\" means a breach of security leading to the accidental or unlawful "
        "destruction, loss, alteration or unauthorised disclosure of, or access to, Personal "
        "Data.",
    ]),
    ("3. Processing Instructions", [
        "Nimbus processes Personal Data only on the documented instructions of the Customer, "
        "including the instructions given through the configuration of the platform. If Nimbus "
        "believes an instruction infringes data protection law, it will inform the Customer "
        "within 5 business days and may suspend the affected processing until the instruction "
        "is confirmed or changed.",
    ]),
    ("4. Confidentiality and Personnel", [
        "Every Nimbus employee and contractor with access to Personal Data is bound by a written "
        "confidentiality obligation that continues for 5 years after their engagement ends. "
        "Staff with production access complete security and privacy training when they join "
        "and every 12 months after that, and pass a background check before access is granted.",
    ]),
    ("5. Security Measures", [
        "Nimbus maintains the technical and organisational measures described in Annex A. "
        "These include encryption of Personal Data at rest with AES-256 and in transit with TLS "
        "1.2 or higher, multi-factor authentication for all staff accounts, and a review of "
        "production access rights every quarter. Nimbus may update these measures as long as "
        "the overall level of protection is not reduced.",
    ]),
    ("6. Sub-processors", [
        "The Customer gives general authorisation for Nimbus to engage Sub-processors. The "
        "Sub-processors approved on the effective date are listed in the table below.",
        "@TABLE",
        "Nimbus will notify the Customer of any new Sub-processor at least 30 days before it "
        "begins processing Personal Data. The Customer may object on reasonable data protection "
        "grounds within 14 days of the notice. If the parties cannot resolve the objection, the "
        "Customer may terminate the affected services and receive a pro-rata refund of prepaid "
        "fees for the remaining term.",
    ]),
    ("7. Security Incidents", [
        "Nimbus will notify the Customer without undue delay, and in any event within 48 hours, "
        "after becoming aware of a Security Incident affecting the Customer's Personal Data. "
        "The notice will describe the nature of the incident, the categories and approximate "
        "number of data subjects concerned, the likely consequences, and the measures taken or "
        "proposed. Nimbus will provide updates at least every 24 hours until the incident is "
        "contained.",
    ]),
    ("8. Data Subject Requests", [
        "If Nimbus receives a request from a data subject relating to the Customer's Personal "
        "Data, it will forward the request to the Customer within 3 business days and will not "
        "respond to it directly unless instructed to. Nimbus will help the Customer answer such "
        "requests, and will complete any assistance the Customer asks for within 10 business "
        "days.",
    ]),
    ("9. Audits", [
        "Nimbus will make its most recent SOC 2 Type II report and ISO/IEC 27001 certificate "
        "available to the Customer on request. In addition, the Customer may audit Nimbus's "
        "compliance with this DPA once every 12 months, with at least 30 days' written notice. "
        "Audits take place during business hours, at the Customer's expense, and must not "
        "disrupt Nimbus's operations. An additional audit is permitted after a Security "
        "Incident.",
    ]),
    ("10. International Transfers", [
        "Personal Data stays in the data region the Customer selects. Where Personal Data is "
        "transferred outside the European Economic Area, including to a Sub-processor, the "
        "transfer is governed by the European Commission's Standard Contractual Clauses "
        "(Module 2, controller to processor), which are incorporated into this DPA by "
        "reference.",
    ]),
    ("11. Return and Deletion of Data", [
        "When the agreement ends, the Customer may export its Personal Data through the platform "
        "for 30 days. After that export window closes, Nimbus deletes the Customer's Personal "
        "Data from its production systems within a further 30 days, and from backups within 35 "
        "days of that, unless the law requires Nimbus to keep it. Nimbus will confirm the "
        "deletion in writing on request.",
    ]),
    ("12. Liability and Governing Law", [
        "Each party's total liability under this DPA is limited to the fees the Customer paid "
        "for the Nimbus Analytics platform in the 12 months before the event that gave rise to "
        "the claim. This DPA is governed by the laws of Ireland, and the courts of Dublin have "
        "exclusive jurisdiction over any dispute arising from it.",
    ]),
]

SUBPROCESSORS = [
    ("Sub-processor", "Purpose", "Location"),
    ("Amazon Web Services", "Cloud hosting and storage", "EU, US, APAC"),
    ("Stripe", "Payment processing", "United States"),
    ("Zendesk", "Customer support tickets", "European Union"),
    ("Twilio SendGrid", "Transactional email", "United States"),
]

ANNEX = [
    "Access control: single sign-on for staff, multi-factor authentication on every account, "
    "least-privilege roles, and quarterly access reviews.",
    "Encryption: AES-256 at rest; TLS 1.2 or higher in transit; keys rotated every 90 days.",
    "Availability: backups every 6 hours, kept for 35 days, replicated to a second "
    "availability zone in the same region.",
    "Monitoring: centralised audit logs kept for 1 year and reviewed by the security team.",
    "Vulnerability management: critical vulnerabilities patched within 7 days; annual "
    "penetration test by an independent firm.",
]


class Agreement(FPDF):
    def header(self) -> None:
        self.set_font("Helvetica", "I", 8)
        self.set_text_color(110)
        self.cell(0, 8, TITLE, align="L")
        self.ln(12)
        self.set_text_color(0)

    def footer(self) -> None:
        self.set_y(-14)
        self.set_font("Helvetica", "I", 8)
        self.set_text_color(110)
        self.cell(0, 8, f"Confidential - Page {self.page_no()} of {{nb}}", align="C")
        self.set_text_color(0)


def paragraph(pdf: Agreement, text: str) -> None:
    pdf.set_font("Helvetica", size=10.5)
    pdf.multi_cell(0, 5.6, text, **NEXT_LINE)
    pdf.ln(2.5)


def build() -> None:
    pdf = Agreement(format="A4")
    pdf.set_title(TITLE)
    pdf.set_author("Nimbus Labs Ltd.")
    pdf.set_margins(22, 18, 22)
    pdf.set_auto_page_break(auto=True, margin=22)
    pdf.add_page()

    pdf.set_font("Helvetica", "B", 16)
    pdf.multi_cell(0, 8, "Data Processing Agreement", **NEXT_LINE)
    pdf.set_font("Helvetica", size=10)
    pdf.multi_cell(0, 6, "Nimbus Labs Ltd. and the Customer  |  Version 2.1, effective "
                         "1 March 2026", **NEXT_LINE)
    pdf.ln(4)

    for heading, paragraphs in CLAUSES:
        pdf.set_font("Helvetica", "B", 12)
        pdf.multi_cell(0, 7, heading, **NEXT_LINE)
        pdf.ln(1)
        for text in paragraphs:
            if text != "@TABLE":
                paragraph(pdf, text)
                continue
            pdf.set_font("Helvetica", size=9.5)
            with pdf.table(col_widths=(34, 46, 26), line_height=6) as table:
                for cells in SUBPROCESSORS:
                    row = table.row()
                    for cell in cells:
                        row.cell(cell)
            pdf.ln(3)

    pdf.set_font("Helvetica", "B", 12)
    pdf.multi_cell(0, 7, "Annex A - Technical and Organisational Measures", **NEXT_LINE)
    pdf.ln(1)
    for item in ANNEX:
        paragraph(pdf, f"- {item}")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    pdf.output(str(OUT))
    print(f"Wrote {OUT} ({pdf.page_no()} pages)")


if __name__ == "__main__":
    build()
