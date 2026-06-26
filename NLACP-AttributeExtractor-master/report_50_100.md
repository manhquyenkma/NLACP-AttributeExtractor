### ID: 50
**Sentence:** The head of educational testing and quality assurance can download learning materials from the learning management system for all courses.
**Subject:** ["role=head, qualifiers={'type': None, 'rank': 'head', 'status': None, 'office': None, 'department': None, 'negated': False}"]
**Resource:** label=educational_testing, qualifier=the, ref=['testing', 'assurance']

### ID: 51
**Sentence:** Users in the academic affairs office can create and view a course offering in the academic information system.
**Subject:** ['role=user, qualifiers=None']
**Resource:** label=affairs_office, qualifier=the, ref=['office']

### ID: 52
**Sentence:** Users in the academic affairs office can modify and delete a course offering in the academic information system.
**Subject:** ['role=user, qualifiers=None']
**Resource:** label=affairs_office, qualifier=the, ref=['office']

### ID: 53
**Sentence:** The course lecturer can view and comment on a course offering in the academic information system.
**Subject:** ['role=lecturer, qualifiers=None']
**Resource:** label=course_offering, qualifier=the, ref=['offering']

### ID: 54
**Sentence:** The course lecturer can accept or deny a course offering in the academic information system.
**Subject:** ['role=lecturer, qualifiers=None']
**Resource:** label=course_offering, qualifier=the, ref=['offering']

### ID: 55
**Sentence:** A department’s academic coordinator and chair can view and comment on a course offering in the academic information system.
**Subject:** ["role=academic_coordinator, qualifiers={'type': None, 'rank': 'coordinator', 'status': None, 'office': None, 'department': None, 'negated': False}", "role=department_chair, qualifiers={'type': None, 'rank': 'chair', 'status': None, 'office': None, 'department': None, 'negated': False}"]
**Resource:** label=course_offering, qualifier=None, ref=['offering']

### ID: 56
**Sentence:** Civilian students cannot register for courses in the course registration system if they owe tuition fees exceeding the prescribed limit.
**Subject:** ["role=student, qualifiers={'type': 'civilian', 'rank': None, 'status': None, 'office': None, 'department': None, 'negated': True}"]
**Resource:** label=courses, qualifier=None, ref=['courses']

### ID: 57
**Sentence:** Civilian students cannot register for a course in the course registration system if the total number of registered credits exceeds the prescribed limit.
**Subject:** ["role=student, qualifiers={'type': 'civilian', 'rank': None, 'status': None, 'office': None, 'department': None, 'negated': True}"]
**Resource:** label=course, qualifier=None, ref=['course']

### ID: 58
**Sentence:** Civilian students cannot register for a course in the course registration system if they have not passed the prerequisites for that course.
**Subject:** ["role=student, qualifiers={'type': 'civilian', 'rank': None, 'status': None, 'office': None, 'department': None, 'negated': True}"]
**Resource:** label=course, qualifier=None, ref=['course']

### ID: 59
**Sentence:** Civilian students cannot register for two courses in the course registration system if the courses have overlapping class schedules.
**Subject:** ["role=student, qualifiers={'type': 'civilian', 'rank': None, 'status': None, 'office': None, 'department': None, 'negated': True}"]
**Resource:** label=courses, qualifier=None, ref=['courses']

### ID: 60
**Sentence:** Civilian students can only register for courses in the course registration system which are currently open.
**Subject:** ["role=student, qualifiers={'type': 'civilian', 'rank': None, 'status': None, 'office': None, 'department': None, 'negated': False}"]
**Resource:** label=courses, qualifier=None, ref=['courses']

### ID: 61
**Sentence:** Civilian students cannot register for courses in the course registration system which are currently closed.
**Subject:** ["role=student, qualifiers={'type': 'civilian', 'rank': None, 'status': None, 'office': None, 'department': None, 'negated': True}"]
**Resource:** label=courses, qualifier=None, ref=['courses']

### ID: 62
**Sentence:** Civilian students cannot register for a course in the course registration system if the number of registered civilian students reaches the prescribed capacity.
**Subject:** ["role=student, qualifiers={'type': 'civilian', 'rank': None, 'status': None, 'office': None, 'department': None, 'negated': True}"]
**Resource:** label=course, qualifier=None, ref=['course']

### ID: 63
**Sentence:** Civilian students can cancel their registration in the course registration system for a course which they have previously registered for successfully.
**Subject:** ["role=student, qualifiers={'type': 'civilian', 'rank': None, 'status': None, 'office': None, 'department': None, 'negated': False}"]
**Resource:** label=registration, qualifier=the, ref=['registration']

### ID: 64
**Sentence:** Civilian students cannot cancel their registration in the course registration system for a course which is currently closed.
**Subject:** ["role=student, qualifiers={'type': 'civilian', 'rank': None, 'status': None, 'office': None, 'department': None, 'negated': True}"]
**Resource:** label=registration, qualifier=the, ref=['registration']

### ID: 65
**Sentence:** Civilian students can view the status of registered courses in the course registration system.
**Subject:** ["role=student, qualifiers={'type': 'civilian', 'rank': None, 'status': None, 'office': None, 'department': None, 'negated': False}"]
**Resource:** label=status, qualifier=the, ref=['status']

### ID: 66
**Sentence:** Civilian students cannot register for courses in the course registration system if the system has not been activated.
**Subject:** ["role=student, qualifiers={'type': 'civilian', 'rank': None, 'status': None, 'office': None, 'department': None, 'negated': True}"]
**Resource:** label=courses, qualifier=None, ref=['courses']

### ID: 67
**Sentence:** Users in the academic affairs office can view the course registration results of all civilian students in the course registration system.
**Subject:** ['role=user, qualifiers=None']
**Resource:** label=affairs_office, qualifier=the, ref=['office']

### ID: 68
**Sentence:** Civilian students’ class schedules are only allowed in the academic information system between 7:00 am and 5:25 pm.
**Subject:** []
**Resource:** None

### ID: 69
**Sentence:** Lecturers’ teaching schedules are only allowed in the academic information system between 7:00 am and 5:25 pm.
**Subject:** []
**Resource:** None

### ID: 70
**Sentence:** A department chair can view the course registration results of all civilian students in the academic information system for that department.
**Subject:** ["role=department_chair, qualifiers={'type': None, 'rank': 'chair', 'status': None, 'office': None, 'department': None, 'negated': False}"]
**Resource:** label=course_registration, qualifier=the, ref=['results']

### ID: 71
**Sentence:** A department’s academic coordinator can view the course registration results of all civilian students in the academic information system for that department.
**Subject:** ["role=academic_coordinator, qualifiers={'type': None, 'rank': 'coordinator', 'status': None, 'office': None, 'department': None, 'negated': False}"]
**Resource:** label=course_registration, qualifier=the, ref=['results']

### ID: 72
**Sentence:** Civilian students can retake a course in the course registration system even if the number of registered civilian students has reached the prescribed capacity.
**Subject:** []
**Resource:** None

### ID: 73
**Sentence:** A department’s academic coordinator and instructors can upload and view the details of suggested graduation project topics in the learning management system.
**Subject:** ["role=academic_coordinator, qualifiers={'type': None, 'rank': 'coordinator', 'status': None, 'office': None, 'department': None, 'negated': False}", 'role=instructor, qualifiers=None']
**Resource:** label=details, qualifier=the, ref=['details']

### ID: 74
**Sentence:** Civilian students can register for graduation project topics only when the system is activated.
**Subject:** ["role=student, qualifiers={'type': 'civilian', 'rank': None, 'status': None, 'office': None, 'department': None, 'negated': False}"]
**Resource:** label=graduation_project_topics, qualifier=None, ref=['topics']

### ID: 75
**Sentence:** A graduation project can have at most two instructors.
**Subject:** []
**Resource:** None

### ID: 76
**Sentence:** A civilian student can register for at most one graduation project topic in the learning management system from the list of suggested topics uploaded by an instructor.
**Subject:** ["role=student, qualifiers={'type': 'civilian', 'rank': None, 'status': None, 'office': None, 'department': None, 'negated': False}"]
**Resource:** label=project_topic, qualifier=None, ref=['topic']

### ID: 77
**Sentence:** A civilian student can register for up to three preferred instructors in the learning management system.
**Subject:** ["role=student, qualifiers={'type': 'civilian', 'rank': None, 'status': None, 'office': None, 'department': None, 'negated': False}"]
**Resource:** label=preferred_instructors, qualifier=None, ref=['instructors']

### ID: 78
**Sentence:** Instructors can accept or deny civilian students’ applications for graduation project topics in the learning management system.
**Subject:** ['role=instructor, qualifiers=None']
**Resource:** label=application, qualifier=None, ref=['applications']

### ID: 79
**Sentence:** Civilian students can view the details of graduation project topics in the learning management system which have been uploaded by instructors.
**Subject:** ["role=student, qualifiers={'type': 'civilian', 'rank': None, 'status': None, 'office': None, 'department': None, 'negated': False}"]
**Resource:** label=details, qualifier=the, ref=['details']

### ID: 80
**Sentence:** A department chair can view the details of all graduation project topics in the learning management system for that department.
**Subject:** ["role=department_chair, qualifiers={'type': None, 'rank': 'chair', 'status': None, 'office': None, 'department': None, 'negated': False}"]
**Resource:** label=details, qualifier=the, ref=['details']

### ID: 81
**Sentence:** Users in the academic affairs office can view the details of all graduation project topics in the learning management system.
**Subject:** ['role=user, qualifiers=None']
**Resource:** label=affairs_office, qualifier=the, ref=['office']

### ID: 82
**Sentence:** A department’s academic coordinator and chair can view the list of civilian students in the learning management system who have successfully registered for graduation project topics in that department.
**Subject:** ["role=academic_coordinator, qualifiers={'type': None, 'rank': 'coordinator', 'status': None, 'office': None, 'department': None, 'negated': False}", "role=department_chair, qualifiers={'type': None, 'rank': 'chair', 'status': None, 'office': None, 'department': None, 'negated': False}"]
**Resource:** label=list, qualifier=the, ref=['list']

### ID: 83
**Sentence:** A department’s academic coordinator and chair can view the list of civilian students in the learning management system who belong to that department and have not yet registered for a graduation project topic.
**Subject:** ["role=academic_coordinator, qualifiers={'type': None, 'rank': 'coordinator', 'status': None, 'office': None, 'department': None, 'negated': False}", "role=department_chair, qualifiers={'type': None, 'rank': 'chair', 'status': None, 'office': None, 'department': None, 'negated': False}"]
**Resource:** label=list, qualifier=the, ref=['list']

### ID: 84
**Sentence:** Instructors can modify the details of their suggested graduation project topics in the learning management system.
**Subject:** ['role=instructor, qualifiers=None']
**Resource:** label=details, qualifier=the, ref=['details']

### ID: 85
**Sentence:** Instructors can delete their suggested graduation project topics in the learning management system.
**Subject:** ['role=instructor, qualifiers=None']
**Resource:** label=suggested_project_topics, qualifier=None, ref=['topics']

### ID: 86
**Sentence:** Civilian students cannot register for graduation project topics until the system is activated.
**Subject:** ["role=student, qualifiers={'type': 'civilian', 'rank': None, 'status': None, 'office': None, 'department': None, 'negated': True}"]
**Resource:** label=graduation_project_topics, qualifier=None, ref=['topics']

### ID: 87
**Sentence:** Users in the academic affairs or finance office can view the details of visiting lecturer agreements in the visiting lecturer management system.
**Subject:** ['role=user, qualifiers=None']
**Resource:** label=academic_affairs, qualifier=the, ref=['affairs', 'office']

### ID: 88
**Sentence:** Users who are not in the academic affairs office cannot number visiting lecturer agreements in the visiting lecturer management system.
**Subject:** ["role=user, qualifiers={'type': None, 'rank': None, 'status': None, 'office': None, 'department': None, 'negated': True}"]
**Resource:** label=affairs_office, qualifier=the, ref=['office']

### ID: 89
**Sentence:** Users who are not in the academic affairs or finance office cannot export entire visiting lecturer agreements in the visiting lecturer management system.
**Subject:** ["role=user, qualifiers={'type': None, 'rank': None, 'status': None, 'office': None, 'department': None, 'negated': True}"]
**Resource:** label=academic_affairs, qualifier=the, ref=['affairs', 'office']

### ID: 90
**Sentence:** A department’s academic coordinator and chair can view the details of visiting lecturer agreements in the visiting lecturer management system.
**Subject:** ["role=academic_coordinator, qualifiers={'type': None, 'rank': 'coordinator', 'status': None, 'office': None, 'department': None, 'negated': False}", "role=department_chair, qualifiers={'type': None, 'rank': 'chair', 'status': None, 'office': None, 'department': None, 'negated': False}"]
**Resource:** label=details, qualifier=the, ref=['details']

### ID: 91
**Sentence:** A department’s academic coordinator can export visiting lecturer agreements in the visiting lecturer management system only for that department.
**Subject:** ["role=academic_coordinator, qualifiers={'type': None, 'rank': 'coordinator', 'status': None, 'office': None, 'department': None, 'negated': False}"]
**Resource:** label=agreement, qualifier=None, ref=['agreements']

### ID: 92
**Sentence:** Visiting lecturer agreements for classes are considered valid only after approval by the finance office.
**Subject:** []
**Resource:** None

### ID: 93
**Sentence:** A visiting lecturer agreement can only be issued in the visiting lecturer management system if it has been numbered, validated, and exported.
**Subject:** ["role=visiting_lecturer, qualifiers={'type': 'visiting', 'rank': None, 'status': None, 'office': None, 'department': None, 'negated': False}"]
**Resource:** label=visiting_management_system, qualifier=None, ref=['system']

### ID: 94
**Sentence:** The finance office can approve visiting lecturer classes in the visiting lecturer management system only if the total teaching hours of a visiting lecturer within the year do not exceed the prescribed limit.
**Subject:** ["role=user, qualifiers={'type': None, 'rank': None, 'status': None, 'office': 'finance', 'department': None, 'negated': False}"]
**Resource:** label=visiting_lecturer_class, qualifier=the, ref=['classes']

### ID: 95
**Sentence:** A visiting lecturer who has not yet retired cannot teach more than 280 hours per year.
**Subject:** []
**Resource:** None

### ID: 96
**Sentence:** A retired visiting lecturer cannot teach more than 560 hours per year.
**Subject:** []
**Resource:** None

### ID: 97
**Sentence:** A department’s academic coordinator cannot modify visiting lecturer agreement information in the visiting lecturer management system after approval by the academic affairs office or finance office.
**Subject:** ["role=academic_coordinator, qualifiers={'type': None, 'rank': 'coordinator', 'status': None, 'office': None, 'department': None, 'negated': True}"]
**Resource:** label=agreement, qualifier=None, ref=['information']

### ID: 98
**Sentence:** A department’s academic coordinator can modify a visiting lecturer agreement in the visiting lecturer management system only if the board of directors has rejected its approval.
**Subject:** ["role=academic_coordinator, qualifiers={'type': None, 'rank': 'coordinator', 'status': None, 'office': None, 'department': None, 'negated': False}"]
**Resource:** label=visiting_lecturer_agreement, qualifier=None, ref=['agreement']

### ID: 99
**Sentence:** A valid visiting lecturer agreement must have a contract number in the visiting lecturer management system.
**Subject:** []
**Resource:** None

### ID: 100
**Sentence:** Visiting lecturers can view their issued visiting lecturer agreements in the visiting lecturer management system.
**Subject:** ["role=visiting_lecturer, qualifiers={'type': 'visiting', 'rank': None, 'status': None, 'office': None, 'department': None, 'negated': False}"]
**Resource:** label=agreement, qualifier=the, ref=['agreements']

