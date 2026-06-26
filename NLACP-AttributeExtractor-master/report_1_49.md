### ID: 1
**Sentence:** Civilian students can view their own final grades in the academic information system for courses which they have taken.
**Subject:** ["role=student, qualifiers={'type': 'civilian', 'rank': None, 'status': None, 'office': None, 'department': None, 'negated': False}"]
**Resource:** label=grade, qualifier=the, ref=['grades']

### ID: 2
**Sentence:** Civilian students can view their own scores in the learning management system for multiple-choice exams which they have taken.
**Subject:** ["role=student, qualifiers={'type': 'civilian', 'rank': None, 'status': None, 'office': None, 'department': None, 'negated': False}"]
**Resource:** label=score, qualifier=the, ref=['scores']

### ID: 3
**Sentence:** Lecturers can view the scores of all civilian students in the learning management system for courses which they teach.
**Subject:** ['role=lecturer, qualifiers=None']
**Resource:** label=score, qualifier=the, ref=['scores']

### ID: 4
**Sentence:** Users in the educational testing and quality assurance office can view the scores of all civilian students in the learning management system for all courses.
**Subject:** ['role=user, qualifiers=None']
**Resource:** label=assurance_office, qualifier=the, ref=['office']

### ID: 5
**Sentence:** Users in the academic affairs office can view the scores of all civilian students in the learning management system for all courses.
**Subject:** ['role=user, qualifiers=None']
**Resource:** label=affairs_office, qualifier=the, ref=['office']

### ID: 6
**Sentence:** The course lecturer can add and view scores in the learning management system for that course.
**Subject:** ['role=lecturer, qualifiers=None']
**Resource:** label=score, qualifier=the, ref=['scores']

### ID: 7
**Sentence:** Course lecturers can update scores and assign final grades in the learning management system for their courses with the approval of the department chair and the head of educational testing and quality assurance.
**Subject:** ['role=lecturer, qualifiers=None']
**Resource:** label=score, qualifier=None, ref=['scores']

### ID: 8
**Sentence:** A department’s academic coordinator can view the scores of all civilian students in the learning management system for that department’s courses.
**Subject:** ["role=academic_coordinator, qualifiers={'type': None, 'rank': 'coordinator', 'status': None, 'office': None, 'department': None, 'negated': False}"]
**Resource:** label=score, qualifier=the, ref=['scores']

### ID: 9
**Sentence:** A department chair can view the scores of all civilian students in the learning management system for that department’s courses.
**Subject:** ["role=department_chair, qualifiers={'type': None, 'rank': 'chair', 'status': None, 'office': None, 'department': None, 'negated': False}"]
**Resource:** label=score, qualifier=the, ref=['scores']

### ID: 10
**Sentence:** Users in the educational testing and quality assurance office can modify civilian students’ final grades in the academic information system with the approval of the head of educational testing and quality assurance.
**Subject:** ['role=user, qualifiers=None']
**Resource:** label=assurance_office, qualifier=the, ref=['office']

### ID: 11
**Sentence:** Civilian students can view rosters in the learning management system for courses which they are taking.
**Subject:** ["role=student, qualifiers={'type': 'civilian', 'rank': None, 'status': None, 'office': None, 'department': None, 'negated': False}"]
**Resource:** label=roster, qualifier=None, ref=['rosters']

### ID: 12
**Sentence:** The course lecturer can view and download the course roster in the learning management system for that course.
**Subject:** ['role=lecturer, qualifiers=None']
**Resource:** label=roster, qualifier=own, ref=['roster']

### ID: 13
**Sentence:** A department’s academic coordinator can view and download rosters in the academic information and learning management systems for that department’s courses.
**Subject:** ["role=academic_coordinator, qualifiers={'type': None, 'rank': 'coordinator', 'status': None, 'office': None, 'department': None, 'negated': False}"]
**Resource:** label=roster, qualifier=own, ref=['rosters']

### ID: 14
**Sentence:** A department chair can view and download rosters in the academic information and learning management systems for that department’s courses.
**Subject:** ["role=department_chair, qualifiers={'type': None, 'rank': 'chair', 'status': None, 'office': None, 'department': None, 'negated': False}"]
**Resource:** label=roster, qualifier=own, ref=['rosters']

### ID: 15
**Sentence:** Users in the academic affairs office can view and modify all rosters in the academic information and learning management systems.
**Subject:** ['role=user, qualifiers=None']
**Resource:** label=affairs_office, qualifier=the, ref=['office']

### ID: 16
**Sentence:** Users in the academic affairs office can upload and download course rosters in the academic information and learning management systems.
**Subject:** ['role=user, qualifiers=None']
**Resource:** label=affairs_office, qualifier=the, ref=['office']

### ID: 17
**Sentence:** Users in the academic affairs office can delete course rosters in the academic information and learning management systems.
**Subject:** ['role=user, qualifiers=None']
**Resource:** label=affairs_office, qualifier=the, ref=['office']

### ID: 18
**Sentence:** Civilian students can view and download their own transcripts in the academic information system.
**Subject:** ["role=student, qualifiers={'type': 'civilian', 'rank': None, 'status': None, 'office': None, 'department': None, 'negated': False}"]
**Resource:** label=transcript, qualifier=own, ref=['transcripts']

### ID: 19
**Sentence:** A department chair can view and download the transcripts of all civilian students in the academic information system for that department.
**Subject:** ["role=department_chair, qualifiers={'type': None, 'rank': 'chair', 'status': None, 'office': None, 'department': None, 'negated': False}"]
**Resource:** label=transcript, qualifier=own, ref=['transcripts']

### ID: 20
**Sentence:** Users in the academic affairs office can view and download the transcripts of all civilian students in the academic information system.
**Subject:** ['role=user, qualifiers=None']
**Resource:** label=affairs_office, qualifier=the, ref=['office']

### ID: 21
**Sentence:** Users in the educational testing and quality assurance office can view and download the transcripts of all civilian students in the academic information system.
**Subject:** ['role=user, qualifiers=None']
**Resource:** label=assurance_office, qualifier=the, ref=['office']

### ID: 22
**Sentence:** The head of academic affairs can modify a civilian student’s transcript in the academic information system with the director’s approval.
**Subject:** ["role=head, qualifiers={'type': None, 'rank': 'head', 'status': None, 'office': None, 'department': None, 'negated': False}"]
**Resource:** label=academic_affairs, qualifier=the, ref=['affairs']

### ID: 23
**Sentence:** The reason for a transcript revision must be recorded in the academic information system after the head of academic affairs modifies a civilian student’s transcript.
**Subject:** ['role=user, qualifiers=None']
**Resource:** label=transcript, qualifier=the, ref=['revision']

### ID: 24
**Sentence:** Applicants can check the status of their own applications in the academic information system.
**Subject:** ['role=applicant, qualifiers=None']
**Resource:** label=status, qualifier=the, ref=['status']

### ID: 25
**Sentence:** Users in the academic affairs office can view and update the status of all applications in the academic information system.
**Subject:** ['role=user, qualifiers=None']
**Resource:** label=affairs_office, qualifier=the, ref=['office']

### ID: 26
**Sentence:** Applicants can modify applications in the academic information system.
**Subject:** ['role=applicant, qualifiers=None']
**Resource:** label=application, qualifier=None, ref=['applications']

### ID: 27
**Sentence:** Applicants can upload and download applications in the academic information system.
**Subject:** ['role=applicant, qualifiers=None']
**Resource:** label=academic_information_system, qualifier=own, ref=['system']

### ID: 28
**Sentence:** Applicants can delete applications in the academic information system only if the applications are currently pending.
**Subject:** ['role=applicant, qualifiers=None']
**Resource:** label=application, qualifier=None, ref=['applications']

### ID: 29
**Sentence:** The course lecturer can create multiple-choice exams and view their details in the learning management system.
**Subject:** ['role=lecturer, qualifiers=None']
**Resource:** label=exam, qualifier=the, ref=['exams']

### ID: 30
**Sentence:** The course lecturer can modify and delete multiple-choice exams in the learning management system.
**Subject:** ['role=lecturer, qualifiers=None']
**Resource:** label=learning_management_system, qualifier=the, ref=['system']

### ID: 31
**Sentence:** The course lecturer can upload and download multiple-choice exams in the learning management system.
**Subject:** ['role=lecturer, qualifiers=None']
**Resource:** label=exam, qualifier=own, ref=['exams']

### ID: 32
**Sentence:** A department’s academic coordinator can view and download multiple-choice exams in the learning management system for that department’s courses.
**Subject:** ["role=academic_coordinator, qualifiers={'type': None, 'rank': 'coordinator', 'status': None, 'office': None, 'department': None, 'negated': False}"]
**Resource:** label=exam, qualifier=own, ref=['exams']

### ID: 33
**Sentence:** A department chair can view and assess multiple-choice exams in the learning management system for that department’s courses.
**Subject:** ["role=department_chair, qualifiers={'type': None, 'rank': 'chair', 'status': None, 'office': None, 'department': None, 'negated': False}"]
**Resource:** label=exam, qualifier=None, ref=['exams']

### ID: 34
**Sentence:** A department chair can upload and download multiple-choice exams in the learning management system for that department’s courses.
**Subject:** ["role=department_chair, qualifiers={'type': None, 'rank': 'chair', 'status': None, 'office': None, 'department': None, 'negated': False}"]
**Resource:** label=exam, qualifier=own, ref=['exams']

### ID: 35
**Sentence:** The head of academic affairs can view and assess multiple-choice exams in the learning management system for all courses.
**Subject:** ["role=head, qualifiers={'type': None, 'rank': 'head', 'status': None, 'office': None, 'department': None, 'negated': False}"]
**Resource:** label=academic_affairs, qualifier=the, ref=['affairs']

### ID: 36
**Sentence:** The head of academic affairs can download multiple-choice exams from the learning management system for all courses.
**Subject:** ["role=head, qualifiers={'type': None, 'rank': 'head', 'status': None, 'office': None, 'department': None, 'negated': False}"]
**Resource:** label=academic_affairs, qualifier=the, ref=['affairs']

### ID: 37
**Sentence:** The head of educational testing and quality assurance can view and assess multiple-choice exams in the learning management system for all courses.
**Subject:** ["role=head, qualifiers={'type': None, 'rank': 'head', 'status': None, 'office': None, 'department': None, 'negated': False}"]
**Resource:** label=educational_testing, qualifier=the, ref=['testing', 'assurance']

### ID: 38
**Sentence:** The head of educational testing and quality assurance can download multiple-choice exams from the learning management system for all courses.
**Subject:** ["role=head, qualifiers={'type': None, 'rank': 'head', 'status': None, 'office': None, 'department': None, 'negated': False}"]
**Resource:** label=educational_testing, qualifier=the, ref=['testing', 'assurance']

### ID: 39
**Sentence:** The course lecturer can create and view learning materials in the learning management system for that course.
**Subject:** ['role=lecturer, qualifiers=None']
**Resource:** label=learning_material, qualifier=the, ref=['materials']

### ID: 40
**Sentence:** The course lecturer can upload and download learning materials in the learning management system.
**Subject:** ['role=lecturer, qualifiers=None']
**Resource:** label=learning_material, qualifier=own, ref=['materials']

### ID: 41
**Sentence:** The course lecturer can modify and delete learning materials in the learning management system.
**Subject:** ['role=lecturer, qualifiers=None']
**Resource:** label=learning_management_system, qualifier=the, ref=['system']

### ID: 42
**Sentence:** Civilian students can view and download learning materials in the learning management system for courses which they are taking.
**Subject:** ["role=student, qualifiers={'type': 'civilian', 'rank': None, 'status': None, 'office': None, 'department': None, 'negated': False}"]
**Resource:** label=learning_material, qualifier=own, ref=['materials']

### ID: 43
**Sentence:** Civilian students can comment on learning materials in the learning management system for courses which they are taking.
**Subject:** ["role=student, qualifiers={'type': 'civilian', 'rank': None, 'status': None, 'office': None, 'department': None, 'negated': False}"]
**Resource:** label=learning_material, qualifier=None, ref=['materials']

### ID: 44
**Sentence:** A department’s academic coordinator can view and download learning materials in the learning management system for that department’s courses.
**Subject:** ["role=academic_coordinator, qualifiers={'type': None, 'rank': 'coordinator', 'status': None, 'office': None, 'department': None, 'negated': False}"]
**Resource:** label=learning_material, qualifier=own, ref=['materials']

### ID: 45
**Sentence:** A department chair can view and assess learning materials in the learning management system for that department’s courses.
**Subject:** ["role=department_chair, qualifiers={'type': None, 'rank': 'chair', 'status': None, 'office': None, 'department': None, 'negated': False}"]
**Resource:** label=learning_material, qualifier=None, ref=['materials']

### ID: 46
**Sentence:** A department chair can upload and download learning materials in the learning management system for that department’s courses.
**Subject:** ["role=department_chair, qualifiers={'type': None, 'rank': 'chair', 'status': None, 'office': None, 'department': None, 'negated': False}"]
**Resource:** label=learning_material, qualifier=own, ref=['materials']

### ID: 47
**Sentence:** The head of academic affairs can view and assess learning materials in the learning management system for all courses.
**Subject:** ["role=head, qualifiers={'type': None, 'rank': 'head', 'status': None, 'office': None, 'department': None, 'negated': False}"]
**Resource:** label=academic_affairs, qualifier=the, ref=['affairs']

### ID: 48
**Sentence:** The head of academic affairs can download learning materials from the learning management system for all courses.
**Subject:** ["role=head, qualifiers={'type': None, 'rank': 'head', 'status': None, 'office': None, 'department': None, 'negated': False}"]
**Resource:** label=academic_affairs, qualifier=the, ref=['affairs']

### ID: 49
**Sentence:** The head of educational testing and quality assurance can view and assess learning materials in the learning management system for all courses.
**Subject:** ["role=head, qualifiers={'type': None, 'rank': 'head', 'status': None, 'office': None, 'department': None, 'negated': False}"]
**Resource:** label=educational_testing, qualifier=the, ref=['testing', 'assurance']

