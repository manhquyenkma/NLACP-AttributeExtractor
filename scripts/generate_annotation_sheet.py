"""
generate_annotation_sheet.py
Tạo file CSV để upload vào Google Sheets cho annotation.
- Lấy câu từ combined_env.json (LitroACP) + self-created sentences
- Xuất CSV với columns chuẩn cho annotation nhóm
Chạy: python data/generate_annotation_sheet.py
"""
import json
import csv
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# ── 100 câu tự tạo theo template ─────────────────────────────────
SELF_CREATED = [
    # === TEMPORAL (50 câu) ===
    # Healthcare
    "A nurse can view patient records during night shifts.",
    "A doctor may access emergency data only between 8pm and 6am.",
    "Senior physicians can update medical records during working hours.",
    "Lab technicians can run tests only within business hours.",
    "On-call nurses may override medication schedules at night.",
    "A pharmacist can issue controlled substances only during daytime.",
    "ICU staff may monitor critical patients during all hours.",
    "A surgeon can access operation logs only during business days.",
    "A hospital administrator can approve budgets before the fiscal deadline.",
    "Interns can access teaching resources only during supervision hours.",
    # Education
    "A professor can submit grades only between semester start and end.",
    "Students may access exam results after the grading period.",
    "Teaching assistants can modify course content during weekdays.",
    "Department heads can approve courses within the registration period.",
    "Librarians can extend borrowing periods during exam weeks.",
    "Students can submit assignments only before the deadline.",
    "A dean can modify academic policies during the review period.",
    "Research assistants can access lab equipment on weekdays.",
    "Advisors may schedule appointments only during office hours.",
    "Students can register for courses only during enrollment periods.",
    # Conference
    "Reviewers can submit evaluations before the review deadline.",
    "Authors may revise papers within the rebuttal period.",
    "PC chairs can assign papers during the bidding phase.",
    "Program committee members can see all reviews after decisions.",
    "Workshop organizers can add sessions before the schedule freeze.",
    "Presenters can upload slides during the week before the conference.",
    "Attendees can access proceedings after the conference date.",
    "Authors of accepted papers can submit camera-ready after notification.",
    "Reviewers may update scores during the discussion phase.",
    "Session chairs can modify room assignments before the conference.",
    # Enterprise
    "Employees can submit expense reports during the monthly cycle.",
    "Managers can approve budgets only within the quarterly review period.",
    "Auditors can access financial records during the audit period.",
    "System administrators can schedule maintenance only during off-hours.",
    "Contractors can access the system only between 9am and 5pm.",
    "Security officers can review access logs during business hours.",
    "Finance staff can process payments only on business days.",
    "HR managers can update payroll only before the payroll deadline.",
    "Interns can access resources only during their contract period.",
    "Executives can view board materials only before the board meeting.",
    # Mixed
    "A senior nurse on night shift may change approved lab procedures.",
    "Junior doctors can consult specialists only during rounds.",
    "A researcher can query the database only during allocated slots.",
    "Lab staff can access samples only during designated testing periods.",
    "A student can take the exam only within the scheduled time window.",
    "A customer can place orders between 8am and 10pm.",
    "A driver can update delivery status only during active routes.",
    "A cashier can process refunds only during business hours.",
    "An IT technician can deploy updates only during maintenance windows.",
    "A compliance officer can freeze accounts only during investigations.",
    # === SPATIAL (50 câu) ===
    # Network/Remote
    "Administrators accessing from internal VPN can modify system settings.",
    "Doctors connecting via the hospital network can view all records.",
    "Remote staff using secure connections may access HR systems.",
    "Users on the corporate network can submit expense reports.",
    "External consultants accessing through the client portal can view reports.",
    "Staff connecting via hospital intranet can update patient records.",
    "Employees working remotely through VPN can access internal tools.",
    "A developer accessing from the office network can deploy code.",
    "A contractor using the company VPN can access project files.",
    "An analyst connecting from the secure network can run queries.",
    # Physical location
    "Nurses at Ward A can access ward-specific patient records.",
    "Physicians within the ICU can override standard protocols.",
    "Staff inside the secure facility can access classified data.",
    "Managers at headquarters can approve regional requests.",
    "Technicians on-site can modify equipment configurations.",
    "Security guards within the data center can access surveillance feeds.",
    "Researchers inside the laboratory can run experiments.",
    "Healthcare workers within the hospital can view patient charts.",
    "A janitor inside the server room must log their presence.",
    "Visitors at reception must register before accessing the building.",
    # Device/Channel
    "Administrators using trusted workstations can modify system settings.",
    "Doctors with hospital-issued devices can access patient records.",
    "Staff using encrypted laptops may work with sensitive data.",
    "Managers via the official management system can approve budgets.",
    "Nurses through the bedside terminal can update patient charts.",
    "Analysts using secure workstations can access financial models.",
    "Employees via two-factor authenticated devices can access HR portals.",
    "Researchers using approved lab systems can record experiment data.",
    "IT staff through the admin console can view server logs.",
    "A user via a managed device can access cloud resources.",
    # Organization unit
    "Employees within the finance department can view budget reports.",
    "Staff within the marketing division can edit campaign materials.",
    "Users within the engineering team can access source code.",
    "Managers within regional offices can approve local expenses.",
    "Members within the security group can access audit trails.",
    "Analysts from the data science department can run ML models.",
    "Admins from the IT department can reset user passwords.",
    "Staff from the legal department can access contract templates.",
    "Members within the executive committee can view board resolutions.",
    "Employees within the HR department can update personnel records.",
    # Mixed
    "A nurse from the ICU can access critical patient data from the hospital network.",
    "A manager within the finance department can approve using the ERP system.",
    "Doctors at City Hospital using hospital devices can view lab results.",
    "Remote employees via VPN within business hours can access the intranet.",
    "Staff inside the headquarters using secure terminals can modify policies.",
    "Administrators within the IT department via the admin portal can reset accounts.",
    "Researchers inside the lab using approved workstations can run simulations.",
    "Nurses at night shift using bedside terminals can update medication records.",
    "Users from the approved network within the corporate premises can login.",
    "An analyst inside the data center using a secure system can access raw logs.",
]


def load_combined_env():
    path = os.path.join(BASE_DIR, "data", "annotated", "combined_env.json")
    if not os.path.exists(path):
        return []
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def build_csv_rows(combined_data, self_created):
    rows = []
    counter = 1

    # === Phần 1: LitroACP sentences có Condition labels ===
    for item in combined_data:
        env_attrs = item.get("env_attributes", [])
        if not env_attrs:
            continue   # chỉ lấy câu đã có env-att

        temporal_gold = "; ".join(
            a["value"] for a in env_attrs if a["category"] == "temporal"
        )
        spatial_gold = "; ".join(
            a["value"] for a in env_attrs if a["category"] == "spatial"
        )

        rows.append({
            "ID":              f"L{counter:04d}",
            "Source":          item.get("dataset", "LitroACP"),
            "Sentence":        item["sentence"],
            "temporal_gold":   temporal_gold,        # pre-filled từ LitroACP
            "spatial_gold":    spatial_gold,
            "temporal_OK":     "CHECK",              # annotator review
            "spatial_OK":      "CHECK",
            "temporal_final":  temporal_gold,        # annotator sửa ở đây
            "spatial_final":   spatial_gold,
            "note":            "",
            "annotator":       "",
            "status":          "pending"
        })
        counter += 1

    # === Phần 2: Self-created sentences ===
    for sent in self_created:
        rows.append({
            "ID":              f"S{counter:04d}",
            "Source":          "self_created",
            "Sentence":        sent,
            "temporal_gold":   "",     # trống — tự annotate
            "spatial_gold":    "",
            "temporal_OK":     "",
            "spatial_OK":      "",
            "temporal_final":  "",     # điền vào đây
            "spatial_final":   "",
            "note":            "",
            "annotator":       "",
            "status":          "pending"
        })
        counter += 1

    return rows


def save_csv(rows, output_path):
    fieldnames = [
        "ID", "Source", "Sentence",
        "temporal_gold", "spatial_gold",
        "temporal_OK", "spatial_OK",
        "temporal_final", "spatial_final",
        "note", "annotator", "status"
    ]
    with open(output_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main():
    print("\n" + "="*55)
    print("  Generate Annotation Sheet for Google Sheets")
    print("="*55)

    combined  = load_combined_env()
    rows      = build_csv_rows(combined, SELF_CREATED)

    out_path  = os.path.join(BASE_DIR, "data", "annotation_sheet.csv")
    save_csv(rows, out_path)

    litro_count = sum(1 for r in rows if r["Source"] != "self_created")
    self_count  = sum(1 for r in rows if r["Source"] == "self_created")

    print(f"\n  LitroACP rows (pre-labeled): {litro_count}")
    print(f"  Self-created rows:           {self_count}")
    print(f"  Total rows:                  {len(rows)}")
    print(f"\n  Saved: {out_path}")
    print("\n  === HUONG DAN UPLOAD GOOGLE SHEETS ===")
    print("  1. Mo Google Sheets moi")
    print("  2. File > Import > Upload CSV nay")
    print("  3. Separator: Comma, dong 1 la header")
    print("  4. Chia se link cho ca nhom (Editor access)")
    print("\n  === QUY TAC ANNOTATION ===")
    print("  - temporal_gold: LitroACP da goi y, REVIEW lai")
    print("  - spatial_gold:  LitroACP da goi y, REVIEW lai")
    print("  - temporal_final / spatial_final: GIA TRI CHINH XAC")
    print("  - status: pending -> done / uncertain / skip")
    print("  - Muc tieu: 20% overlap (nguoi 1+2 lam cung 1/5 so cau)")
    print("="*55 + "\n")


if __name__ == "__main__":
    main()
