from flask import Flask, render_template, request, jsonify, session, redirect, url_for, flash
import json
import random
import datetime
import uuid
import cv2
import numpy as np
from ultralytics import YOLO
import base64
import io
from PIL import Image
import threading
import time
import os
from werkzeug.utils import secure_filename
from werkzeug.security import generate_password_hash, check_password_hash
from flask_sqlalchemy import SQLAlchemy
from sqlalchemy import func
from datetime import datetime

app = Flask(__name__)
app.secret_key = 'your-secret-key-here-change-this-in-production'

# Database configuration
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///interview_system.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)

# Configure upload folder
UPLOAD_FOLDER = 'uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Initialize YOLO model
try:
    model = YOLO('yolov8n.pt')
except:
    model = None
    print("YOLO model not found. Please download yolov8n.pt")

# Database Models
class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password_hash = db.Column(db.String(128), nullable=False)
    role = db.Column(db.String(20), nullable=False, default='student')  # 'admin' or 'student'
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    is_active = db.Column(db.Boolean, default=True)
    
    def set_password(self, password):
        self.password_hash = generate_password_hash(password)
    
    def check_password(self, password):
        return check_password_hash(self.password_hash, password)

class Candidate(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    phone = db.Column(db.String(20), nullable=False)
    position = db.Column(db.String(100), nullable=False)
    resume = db.Column(db.String(255))  # filename
    added_date = db.Column(db.DateTime, default=datetime.utcnow)
    created_by = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    
    # Relationships
    interviews = db.relationship('Interview', backref='candidate', lazy=True)

class JobRequirement(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    title = db.Column(db.String(100), nullable=False)
    department = db.Column(db.String(100), nullable=False)
    experience = db.Column(db.String(50), nullable=False)
    skills = db.Column(db.Text, nullable=False)
    description = db.Column(db.Text, nullable=False)
    created_date = db.Column(db.DateTime, default=datetime.utcnow)
    created_by = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    is_active = db.Column(db.Boolean, default=True)
    
    # Relationships
    interviews = db.relationship('Interview', backref='job', lazy=True)

class Interview(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    candidate_id = db.Column(db.Integer, db.ForeignKey('candidate.id'), nullable=False)
    job_id = db.Column(db.Integer, db.ForeignKey('job_requirement.id'), nullable=False)
    date = db.Column(db.Date, nullable=False)
    time = db.Column(db.Time, nullable=False)
    type = db.Column(db.String(20), nullable=False)  # 'technical', 'behavioral', 'both'
    status = db.Column(db.String(20), default='Scheduled')  # 'Scheduled', 'Completed', 'Cancelled'
    created_date = db.Column(db.DateTime, default=datetime.utcnow)
    created_by = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    
    # Interview results
    session_id = db.Column(db.String(36), unique=True)
    technical_score = db.Column(db.Integer)
    behavioral_score = db.Column(db.Integer)
    integrity_score = db.Column(db.Integer)
    overall_score = db.Column(db.Float)
    cheating_violations = db.Column(db.Integer, default=0)
    tab_changes = db.Column(db.Integer, default=0)
    recommendation = db.Column(db.String(100))
    completed_at = db.Column(db.DateTime)

class InterviewSession(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    session_id = db.Column(db.String(36), unique=True, nullable=False)
    interview_id = db.Column(db.Integer, db.ForeignKey('interview.id'))
    start_time = db.Column(db.DateTime, default=datetime.utcnow)
    end_time = db.Column(db.DateTime)
    status = db.Column(db.String(20), default='active')  # 'active', 'completed', 'terminated'
    current_question_index = db.Column(db.Integer, default=0)
    tab_changes = db.Column(db.Integer, default=0)
    frame_counter = db.Column(db.Integer, default=0)

class InterviewQuestion(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    session_id = db.Column(db.String(36), db.ForeignKey('interview_session.session_id'), nullable=False)
    question = db.Column(db.Text, nullable=False)
    question_type = db.Column(db.String(20), nullable=False)  # 'technical', 'behavioral'
    question_number = db.Column(db.Integer, nullable=False)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)

class InterviewAnswer(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    session_id = db.Column(db.String(36), db.ForeignKey('interview_session.session_id'), nullable=False)
    question_id = db.Column(db.Integer, db.ForeignKey('interview_question.id'), nullable=False)
    answer = db.Column(db.Text, nullable=False)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)

class CheatingViolation(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    session_id = db.Column(db.String(36), db.ForeignKey('interview_session.session_id'), nullable=False)
    violation_type = db.Column(db.String(50), nullable=False)  # 'object_detected', 'multiple_persons', 'tab_change'
    object_name = db.Column(db.String(50))  # for object detection
    confidence = db.Column(db.Float)  # for object detection
    person_count = db.Column(db.Integer)  # for multiple persons
    image_path = db.Column(db.String(255))  # saved image path
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)

# Sample interview questions
INTERVIEW_QUESTIONS = {
    "technical": [
        "What is the difference between Python lists and tuples?",
        "Explain the concept of Object-Oriented Programming.",
        "What is a database index and why is it important?",
        "Describe the difference between GET and POST HTTP methods.",
        "What is the time complexity of binary search?",
        "Explain what is meant by 'Big O' notation.",
        "What is the difference between SQL and NoSQL databases?",
        "Describe the concept of machine learning.",
        "What is version control and why is it important?",
        "Explain the difference between frontend and backend development."
    ],
    "behavioral": [
        "Tell me about a challenging project you worked on.",
        "How do you handle tight deadlines?",
        "Describe a time when you had to work with a difficult team member.",
        "What motivates you in your work?",
        "How do you stay updated with new technologies?",
        "Tell me about a mistake you made and how you handled it.",
        "Describe your ideal work environment.",
        "How do you prioritize tasks when everything seems urgent?",
        "Tell me about a time you had to learn something new quickly.",
        "What are your long-term career goals?"
    ]
}

# Cheating detection classes
CHEATING_OBJECTS = ['cell phone', 'book', 'laptop', 'tablet']
PERSON_CLASS = 'person'

# Helper functions
def init_db():
    """Initialize database with tables and admin user"""
    with app.app_context():
        db.create_all()
        
        # Create admin user if not exists
        admin = User.query.filter_by(email='admin@gmail.com').first()
        if not admin:
            admin = User(
                email='admin@gmail.com',
                role='admin'
            )
            admin.set_password('admin123')
            db.session.add(admin)
        
        # Create sample student user if not exists
        student = User.query.filter_by(email='student@gmail.com').first()
        if not student:
            student = User(
                email='student@gmail.com',
                role='student'
            )
            student.set_password('student123')
            db.session.add(student)
        
        db.session.commit()
        print("Database initialized successfully!")

def validate_email(email):
    """Simple email validation"""
    import re
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    return re.match(pattern, email) is not None

def validate_phone(phone):
    """Simple phone validation"""
    import re
    pattern = r'^[+]?[\d\s\-\(\)]{10,}$'
    return re.match(pattern, phone) is not None

# Routes
@app.route('/')
def home():
    return render_template('home.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form.get('email', '').strip()
        password = request.form.get('password', '')
        
        if not email or not password:
            flash('Email and password are required.', 'error')
            return render_template('login.html')
        
        if not validate_email(email):
            flash('Please enter a valid email address.', 'error')
            return render_template('login.html')
        
        user = User.query.filter_by(email=email, is_active=True).first()
        
        if user and user.check_password(password):
            session['user_id'] = user.id
            session['user_email'] = user.email
            session['user_role'] = user.role
            
            if user.role == 'admin':
                return redirect(url_for('dashboard'))
            else:
                return redirect(url_for('interview'))
        else:
            flash('Invalid email or password.', 'error')
    
    return render_template('login.html')

@app.route('/logout')
def logout():
    session.clear()
    flash('You have been logged out successfully.', 'info')
    return redirect(url_for('home'))

@app.route('/dashboard')
def dashboard():
    if 'user_id' not in session or session.get('user_role') != 'admin':
        flash('Access denied. Admin privileges required.', 'error')
        return redirect(url_for('login'))
    
    candidates = Candidate.query.all()
    jobs = JobRequirement.query.filter_by(is_active=True).all()
    interviews = Interview.query.all()
    
    return render_template('dashboard.html', 
                         candidates=candidates, 
                         jobs=jobs, 
                         interviews=interviews)

@app.route('/add_candidate', methods=['GET', 'POST'])
def add_candidate():
    if 'user_id' not in session or session.get('user_role') != 'admin':
        flash('Access denied. Admin privileges required.', 'error')
        return redirect(url_for('login'))
    
    if request.method == 'POST':
        name = request.form.get('name', '').strip()
        email = request.form.get('email', '').strip()
        phone = request.form.get('phone', '').strip()
        position = request.form.get('position', '').strip()
        
        # Validation
        errors = []
        if not name or len(name) < 2:
            errors.append('Name must be at least 2 characters long.')
        if not email or not validate_email(email):
            errors.append('Please enter a valid email address.')
        if not phone or not validate_phone(phone):
            errors.append('Please enter a valid phone number.')
        if not position or len(position) < 2:
            errors.append('Position must be at least 2 characters long.')
        
        # Check if email already exists
        if email and Candidate.query.filter_by(email=email).first():
            errors.append('A candidate with this email already exists.')
        
        if errors:
            for error in errors:
                flash(error, 'error')
            return render_template('add_candidate.html')
        
        # Handle file upload
        resume = request.files.get('resume')
        resume_filename = None
        if resume and resume.filename:
            resume_filename = secure_filename(resume.filename)
            resume.save(os.path.join(app.config['UPLOAD_FOLDER'], resume_filename))
        
        candidate = Candidate(
            name=name,
            email=email,
            phone=phone,
            position=position,
            resume=resume_filename,
            created_by=session['user_id']
        )
        
        try:
            db.session.add(candidate)
            db.session.commit()
            flash('Candidate added successfully!', 'success')
            return redirect(url_for('dashboard'))
        except Exception as e:
            db.session.rollback()
            flash('Error adding candidate. Please try again.', 'error')
            print(f"Error: {e}")
    
    return render_template('add_candidate.html')

@app.route('/add_job', methods=['GET', 'POST'])
def add_job():
    if 'user_id' not in session or session.get('user_role') != 'admin':
        flash('Access denied. Admin privileges required.', 'error')
        return redirect(url_for('login'))
    
    if request.method == 'POST':
        title = request.form.get('title', '').strip()
        department = request.form.get('department', '').strip()
        experience = request.form.get('experience', '').strip()
        skills = request.form.get('skills', '').strip()
        description = request.form.get('description', '').strip()
        
        # Validation
        errors = []
        if not title or len(title) < 3:
            errors.append('Job title must be at least 3 characters long.')
        if not department or len(department) < 2:
            errors.append('Department must be at least 2 characters long.')
        if not experience:
            errors.append('Experience requirement is required.')
        if not skills or len(skills) < 10:
            errors.append('Skills must be at least 10 characters long.')
        if not description or len(description) < 20:
            errors.append('Job description must be at least 20 characters long.')
        
        if errors:
            for error in errors:
                flash(error, 'error')
            return render_template('add_job.html')
        
        job = JobRequirement(
            title=title,
            department=department,
            experience=experience,
            skills=skills,
            description=description,
            created_by=session['user_id']
        )
        
        try:
            db.session.add(job)
            db.session.commit()
            flash('Job requirement added successfully!', 'success')
            return redirect(url_for('dashboard'))
        except Exception as e:
            db.session.rollback()
            flash('Error adding job requirement. Please try again.', 'error')
            print(f"Error: {e}")
    
    return render_template('add_job.html')

@app.route('/schedule_interview', methods=['GET', 'POST'])
def schedule_interview():
    if 'user_id' not in session or session.get('user_role') != 'admin':
        flash('Access denied. Admin privileges required.', 'error')
        return redirect(url_for('login'))
    
    if request.method == 'POST':
        candidate_id = request.form.get('candidate_id')
        job_id = request.form.get('job_id')
        date = request.form.get('date')
        time = request.form.get('time')
        interview_type = request.form.get('type')
        
        # Validation
        errors = []
        if not candidate_id:
            errors.append('Please select a candidate.')
        if not job_id:
            errors.append('Please select a job position.')
        if not date:
            errors.append('Please select a date.')
        if not time:
            errors.append('Please select a time.')
        if not interview_type:
            errors.append('Please select interview type.')
        
        # Validate date is in future
        if date:
            try:
                interview_date = datetime.strptime(date, '%Y-%m-%d').date()
                if interview_date <= datetime.now().date():
                    errors.append('Interview date must be in the future.')
            except ValueError:
                errors.append('Invalid date format.')
        
        if errors:
            for error in errors:
                flash(error, 'error')
            candidates = Candidate.query.all()
            jobs = JobRequirement.query.filter_by(is_active=True).all()
            return render_template('schedule_interview.html', candidates=candidates, jobs=jobs)
        
        interview = Interview(
            candidate_id=candidate_id,
            job_id=job_id,
            date=datetime.strptime(date, '%Y-%m-%d').date(),
            time=datetime.strptime(time, '%H:%M').time(),
            type=interview_type,
            created_by=session['user_id']
        )
        
        try:
            db.session.add(interview)
            db.session.commit()
            flash('Interview scheduled successfully!', 'success')
            return redirect(url_for('dashboard'))
        except Exception as e:
            db.session.rollback()
            flash('Error scheduling interview. Please try again.', 'error')
            print(f"Error: {e}")
    
    candidates = Candidate.query.all()
    jobs = JobRequirement.query.filter_by(is_active=True).all()
    return render_template('schedule_interview.html', candidates=candidates, jobs=jobs)

@app.route('/interview')
def interview():
    if 'user_id' not in session:
        flash('Please login to access the interview.', 'error')
        return redirect(url_for('login'))
    return render_template('interview.html')

@app.route('/start_interview', methods=['POST'])
def start_interview():
    if 'user_id' not in session:
        return jsonify({'error': 'Unauthorized'}), 401
    
    session_id = str(uuid.uuid4())
    session['session_id'] = session_id
    
    # Create interview session
    interview_session = InterviewSession(
        session_id=session_id,
        start_time=datetime.utcnow()
    )
    
    try:
        db.session.add(interview_session)
        db.session.commit()
        
        # Generate first question
        question_type = random.choice(['technical', 'behavioral'])
        question_text = random.choice(INTERVIEW_QUESTIONS[question_type])
        
        question = InterviewQuestion(
            session_id=session_id,
            question=question_text,
            question_type=question_type,
            question_number=1
        )
        
        db.session.add(question)
        db.session.commit()
        
        return jsonify({
            'status': 'success',
            'session_id': session_id,
            'question': question_text,
            'question_number': 1
        })
        
    except Exception as e:
        db.session.rollback()
        return jsonify({'error': 'Failed to start interview'}), 500

@app.route('/get_next_question', methods=['POST'])
def get_next_question():
    session_id = session.get('session_id')
    if not session_id:
        return jsonify({'error': 'Invalid session'}), 400
    
    interview_session = InterviewSession.query.filter_by(session_id=session_id).first()
    if not interview_session:
        return jsonify({'error': 'Session not found'}), 400
    
    # Store previous answer if provided
    data = request.get_json()
    if data and data.get('answer'):
        # Get the current question
        current_question = InterviewQuestion.query.filter_by(
            session_id=session_id,
            question_number=interview_session.current_question_index + 1
        ).first()
        
        if current_question:
            answer = InterviewAnswer(
                session_id=session_id,
                question_id=current_question.id,
                answer=data['answer']
            )
            db.session.add(answer)
    
    # Update question index
    interview_session.current_question_index += 1
    
    if interview_session.current_question_index >= 5:  # Limit to 5 questions
        interview_session.status = 'completed'
        interview_session.end_time = datetime.utcnow()
        db.session.commit()
        
        return jsonify({
            'status': 'completed',
            'message': 'Interview completed successfully!'
        })
    
    # Generate next question
    question_type = random.choice(['technical', 'behavioral'])
    question_text = random.choice(INTERVIEW_QUESTIONS[question_type])
    
    question = InterviewQuestion(
        session_id=session_id,
        question=question_text,
        question_type=question_type,
        question_number=interview_session.current_question_index + 1
    )
    
    try:
        db.session.add(question)
        db.session.commit()
        
        return jsonify({
            'status': 'success',
            'question': question_text,
            'question_number': interview_session.current_question_index + 1
        })
        
    except Exception as e:
        db.session.rollback()
        return jsonify({'error': 'Failed to generate question'}), 500

@app.route('/submit_answer', methods=['POST'])
def submit_answer():
    session_id = session.get('session_id')
    if not session_id:
        return jsonify({'error': 'Invalid session'}), 400
    
    interview_session = InterviewSession.query.filter_by(session_id=session_id).first()
    if not interview_session:
        return jsonify({'error': 'Session not found'}), 400
    
    data = request.get_json()
    answer_text = data.get('answer', '')
    
    # Get current question
    current_question = InterviewQuestion.query.filter_by(
        session_id=session_id,
        question_number=interview_session.current_question_index + 1
    ).first()
    
    if current_question:
        answer = InterviewAnswer(
            session_id=session_id,
            question_id=current_question.id,
            answer=answer_text
        )
        
        try:
            db.session.add(answer)
            db.session.commit()
            return jsonify({'status': 'success'})
        except Exception as e:
            db.session.rollback()
            return jsonify({'error': 'Failed to save answer'}), 500
    
    return jsonify({'error': 'Question not found'}), 400

@app.route('/detect_cheating', methods=['POST'])
def detect_cheating():
    session_id = session.get('session_id')
    if not session_id:
        return jsonify({'error': 'Invalid session'}), 400

    interview_session = InterviewSession.query.filter_by(session_id=session_id).first()
    if not interview_session:
        return jsonify({'error': 'Session not found'}), 400

    if not model:
        return jsonify({'error': 'YOLO model not loaded'}), 500

    try:
        # Update frame counter
        interview_session.frame_counter += 1
        current_frame = interview_session.frame_counter

        data = request.get_json()
        image_data = data.get('image')

        # Decode base64 image
        image_data = image_data.split(',')[1]
        image_bytes = base64.b64decode(image_data)
        image = Image.open(io.BytesIO(image_bytes))
        image_np = np.array(image)

        # Process only every second frame
        if current_frame % 2 != 0:
            db.session.commit()
            return jsonify({
                'violations': [],
                'person_count': 0,
                'status': 'skipped',
                'frame_number': current_frame
            })

        # Run YOLO detection
        results = model(image_np)

        violations = []
        person_count = 0

        for result in results:
            boxes = result.boxes
            if boxes is not None:
                for box in boxes:
                    class_id = int(box.cls[0])
                    class_name = model.names[class_id]
                    confidence = float(box.conf[0])

                    if confidence > 0.5:
                        if class_name == PERSON_CLASS:
                            person_count += 1
                        elif class_name in CHEATING_OBJECTS:
                            violations.append({
                                'object': class_name,
                                'confidence': confidence,
                                'timestamp': datetime.utcnow().isoformat()
                            })

        # Check for multiple persons
        if person_count > 1:
            violations.append({
                'object': 'multiple_persons',
                'count': person_count,
                'timestamp': datetime.utcnow().isoformat()
            })

        # Save violations to database
        saved_path = None
        if violations:
            # Save violation image
            frames_dir = f"frames/{session_id}"
            os.makedirs(frames_dir, exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
            frame_filename = f"violation_{current_frame:06d}_{timestamp}.jpg"
            saved_path = os.path.join(frames_dir, frame_filename)
            image.save(saved_path, 'JPEG', quality=100)

            # Save each violation
            for violation in violations:
                cheating_violation = CheatingViolation(
                    session_id=session_id,
                    violation_type='object_detected' if violation['object'] != 'multiple_persons' else 'multiple_persons',
                    object_name=violation['object'],
                    confidence=violation.get('confidence'),
                    person_count=violation.get('count'),
                    image_path=saved_path
                )
                db.session.add(cheating_violation)

        db.session.commit()

        return jsonify({
            'violations': violations,
            'person_count': person_count,
            'status': 'violation_detected' if violations else 'clean',
            'frame_number': current_frame,
            'processed': True,
            'saved_path': saved_path
        })

    except Exception as e:
        db.session.rollback()
        print(f"Error in detect_cheating: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/report_tab_change', methods=['POST'])
def report_tab_change():
    session_id = session.get('session_id')
    if not session_id:
        return jsonify({'error': 'Invalid session'}), 400
    
    interview_session = InterviewSession.query.filter_by(session_id=session_id).first()
    if not interview_session:
        return jsonify({'error': 'Session not found'}), 400
    
    interview_session.tab_changes += 1
    
    # Save as violation
    violation = CheatingViolation(
        session_id=session_id,
        violation_type='tab_change'
    )
    
    try:
        db.session.add(violation)
        db.session.commit()
        
        return jsonify({
            'status': 'recorded',
            'total_tab_changes': interview_session.tab_changes
        })
    except Exception as e:
        db.session.rollback()
        return jsonify({'error': 'Failed to record tab change'}), 500

@app.route('/get_interview_results')
def get_interview_results():
    session_id = session.get('session_id')
    if not session_id:
        return jsonify({'error': 'Invalid session'}), 400
    
    interview_session = InterviewSession.query.filter_by(session_id=session_id).first()
    if not interview_session:
        return jsonify({'error': 'Session not found'}), 400
    
    # Get violation counts
    violations = CheatingViolation.query.filter_by(session_id=session_id).all()
    cheating_violations = len([v for v in violations if v.violation_type != 'tab_change'])
    tab_changes = len([v for v in violations if v.violation_type == 'tab_change'])
    
    # Get answers count
    answers_count = InterviewAnswer.query.filter_by(session_id=session_id).count()
    
    # Generate scores
    technical_score = random.randint(60, 95)
    behavioral_score = random.randint(65, 90)
    integrity_score = max(0, 100 - (cheating_violations * 10) - (tab_changes * 5))
    
    overall_score = (technical_score + behavioral_score + integrity_score) / 3
    recommendation = 'Proceed to next round' if overall_score >= 70 else 'Requires further evaluation'
    
    # Update interview session with results
    interview_session.end_time = datetime.utcnow()
    interview_session.status = 'completed'
    
    try:
        db.session.commit()
        
        # Calculate duration
        duration = interview_session.end_time - interview_session.start_time
        
        results = {
            'session_id': session_id,
            'duration': str(duration),
            'questions_answered': answers_count,
            'technical_score': technical_score,
            'behavioral_score': behavioral_score,
            'integrity_score': integrity_score,
            'overall_score': round(overall_score, 2),
            'cheating_violations': cheating_violations,
            'tab_changes': tab_changes,
            'violations_detail': [
                {
                    'type': v.violation_type,
                    'object': v.object_name,
                    'confidence': v.confidence,
                    'person_count': v.person_count,
                    'timestamp': v.timestamp.isoformat()
                } for v in violations
            ],
            'recommendation': recommendation
        }
        
        return jsonify(results)
        
    except Exception as e:
        db.session.rollback()
        return jsonify({'error': 'Failed to generate results'}), 500

# Additional admin routes for managing the system

@app.route('/admin/users')
def admin_users():
    if 'user_id' not in session or session.get('user_role') != 'admin':
        flash('Access denied. Admin privileges required.', 'error')
        return redirect(url_for('login'))
    
    users = User.query.all()
    return render_template('admin_users.html', users=users)

@app.route('/admin/interviews')
def admin_interviews():
    if 'user_id' not in session or session.get('user_role') != 'admin':
        flash('Access denied. Admin privileges required.', 'error')
        return redirect(url_for('login'))
    
    interviews = db.session.query(Interview, Candidate, JobRequirement).join(
        Candidate, Interview.candidate_id == Candidate.id
    ).join(
        JobRequirement, Interview.job_id == JobRequirement.id
    ).all()
    
    return render_template('admin_interviews.html', interviews=interviews)

@app.route('/admin/interview_results/<int:interview_id>')
def admin_interview_results(interview_id):
    if 'user_id' not in session or session.get('user_role') != 'admin':
        flash('Access denied. Admin privileges required.', 'error')
        return redirect(url_for('login'))
    
    interview = Interview.query.get_or_404(interview_id)
    
    if interview.session_id:
        session_data = InterviewSession.query.filter_by(session_id=interview.session_id).first()
        questions = InterviewQuestion.query.filter_by(session_id=interview.session_id).all()
        answers = InterviewAnswer.query.filter_by(session_id=interview.session_id).all()
        violations = CheatingViolation.query.filter_by(session_id=interview.session_id).all()
        
        return render_template('admin_interview_results.html', 
                             interview=interview,
                             session_data=session_data,
                             questions=questions,
                             answers=answers,
                             violations=violations)
    else:
        flash('Interview has not been completed yet.', 'info')
        return redirect(url_for('admin_interviews'))

@app.route('/admin/candidate/<int:candidate_id>')
def admin_candidate_details(candidate_id):
    if 'user_id' not in session or session.get('user_role') != 'admin':
        flash('Access denied. Admin privileges required.', 'error')
        return redirect(url_for('login'))
    
    candidate = Candidate.query.get_or_404(candidate_id)
    candidate_interviews = Interview.query.filter_by(candidate_id=candidate_id).all()
    
    return render_template('admin_candidate_details.html', 
                         candidate=candidate,
                         interviews=candidate_interviews)

@app.route('/admin/job/<int:job_id>')
def admin_job_details(job_id):
    if 'user_id' not in session or session.get('user_role') != 'admin':
        flash('Access denied. Admin privileges required.', 'error')
        return redirect(url_for('login'))
    
    job = JobRequirement.query.get_or_404(job_id)
    job_interviews = Interview.query.filter_by(job_id=job_id).all()
    
    return render_template('admin_job_details.html', 
                         job=job,
                         interviews=job_interviews)

@app.route('/admin/deactivate_job/<int:job_id>')
def admin_deactivate_job(job_id):
    if 'user_id' not in session or session.get('user_role') != 'admin':
        flash('Access denied. Admin privileges required.', 'error')
        return redirect(url_for('login'))
    
    job = JobRequirement.query.get_or_404(job_id)
    job.is_active = False
    
    try:
        db.session.commit()
        flash('Job requirement deactivated successfully.', 'success')
    except Exception as e:
        db.session.rollback()
        flash('Error deactivating job requirement.', 'error')
    
    return redirect(url_for('dashboard'))

@app.route('/admin/activate_job/<int:job_id>')
def admin_activate_job(job_id):
    if 'user_id' not in session or session.get('user_role') != 'admin':
        flash('Access denied. Admin privileges required.', 'error')
        return redirect(url_for('login'))
    
    job = JobRequirement.query.get_or_404(job_id)
    job.is_active = True
    
    try:
        db.session.commit()
        flash('Job requirement activated successfully.', 'success')
    except Exception as e:
        db.session.rollback()
        flash('Error activating job requirement.', 'error')
    
    return redirect(url_for('dashboard'))

@app.route('/admin/update_interview_status/<int:interview_id>/<status>')
def admin_update_interview_status(interview_id, status):
    if 'user_id' not in session or session.get('user_role') != 'admin':
        flash('Access denied. Admin privileges required.', 'error')
        return redirect(url_for('login'))
    
    if status not in ['Scheduled', 'Completed', 'Cancelled']:
        flash('Invalid status.', 'error')
        return redirect(url_for('admin_interviews'))
    
    interview = Interview.query.get_or_404(interview_id)
    interview.status = status
    
    try:
        db.session.commit()
        flash(f'Interview status updated to {status}.', 'success')
    except Exception as e:
        db.session.rollback()
        flash('Error updating interview status.', 'error')
    
    return redirect(url_for('admin_interviews'))

@app.route('/admin/statistics')
def admin_statistics():
    if 'user_id' not in session or session.get('user_role') != 'admin':
        flash('Access denied. Admin privileges required.', 'error')
        return redirect(url_for('login'))
    
    # Get various statistics
    total_candidates = Candidate.query.count()
    total_jobs = JobRequirement.query.filter_by(is_active=True).count()
    total_interviews = Interview.query.count()
    completed_interviews = Interview.query.filter_by(status='Completed').count()
    
    # Get recent activities
    recent_candidates = Candidate.query.order_by(Candidate.added_date.desc()).limit(5).all()
    recent_interviews = Interview.query.order_by(Interview.created_date.desc()).limit(5).all()
    
    # Get violation statistics
    total_violations = CheatingViolation.query.count()
    tab_change_violations = CheatingViolation.query.filter_by(violation_type='tab_change').count()
    object_violations = CheatingViolation.query.filter(CheatingViolation.violation_type != 'tab_change').count()
    
    stats = {
        'total_candidates': total_candidates,
        'total_jobs': total_jobs,
        'total_interviews': total_interviews,
        'completed_interviews': completed_interviews,
        'recent_candidates': recent_candidates,
        'recent_interviews': recent_interviews,
        'total_violations': total_violations,
        'tab_change_violations': tab_change_violations,
        'object_violations': object_violations
    }
    
    return render_template('admin_statistics.html', stats=stats)

@app.route('/admin/create_user', methods=['GET', 'POST'])
def admin_create_user():
    if 'user_id' not in session or session.get('user_role') != 'admin':
        flash('Access denied. Admin privileges required.', 'error')
        return redirect(url_for('login'))
    
    if request.method == 'POST':
        email = request.form.get('email', '').strip()
        password = request.form.get('password', '')
        confirm_password = request.form.get('confirm_password', '')
        role = request.form.get('role', 'student')
        
        # Validation
        errors = []
        if not email or not validate_email(email):
            errors.append('Please enter a valid email address.')
        if not password or len(password) < 6:
            errors.append('Password must be at least 6 characters long.')
        if password != confirm_password:
            errors.append('Passwords do not match.')
        if role not in ['admin', 'student']:
            errors.append('Invalid role selected.')
        
        # Check if user already exists
        if email and User.query.filter_by(email=email).first():
            errors.append('A user with this email already exists.')
        
        if errors:
            for error in errors:
                flash(error, 'error')
            return render_template('admin_create_user.html')
        
        user = User(
            email=email,
            role=role
        )
        user.set_password(password)
        
        try:
            db.session.add(user)
            db.session.commit()
            flash('User created successfully!', 'success')
            return redirect(url_for('admin_users'))
        except Exception as e:
            db.session.rollback()
            flash('Error creating user. Please try again.', 'error')
            print(f"Error: {e}")
    
    return render_template('admin_create_user.html')

@app.route('/admin/deactivate_user/<int:user_id>')
def admin_deactivate_user(user_id):
    if 'user_id' not in session or session.get('user_role') != 'admin':
        flash('Access denied. Admin privileges required.', 'error')
        return redirect(url_for('login'))
    
    # Prevent admin from deactivating themselves
    if user_id == session['user_id']:
        flash('You cannot deactivate your own account.', 'error')
        return redirect(url_for('admin_users'))
    
    user = User.query.get_or_404(user_id)
    user.is_active = False
    
    try:
        db.session.commit()
        flash('User deactivated successfully.', 'success')
    except Exception as e:
        db.session.rollback()
        flash('Error deactivating user.', 'error')
    
    return redirect(url_for('admin_users'))

@app.route('/admin/activate_user/<int:user_id>')
def admin_activate_user(user_id):
    if 'user_id' not in session or session.get('user_role') != 'admin':
        flash('Access denied. Admin privileges required.', 'error')
        return redirect(url_for('login'))
    
    user = User.query.get_or_404(user_id)
    user.is_active = True
    
    try:
        db.session.commit()
        flash('User activated successfully.', 'success')
    except Exception as e:
        db.session.rollback()
        flash('Error activating user.', 'error')
    
    return redirect(url_for('admin_users'))

# Error handlers
@app.errorhandler(404)
def not_found_error(error):
    return render_template('404.html'), 404

@app.errorhandler(500)
def internal_error(error):
    db.session.rollback()
    return render_template('500.html'), 500

# Context processors for templates
@app.context_processor
def inject_user():
    if 'user_id' in session:
        return {
            'current_user': {
                'id': session['user_id'],
                'email': session['user_email'],
                'role': session['user_role']
            }
        }
    return {'current_user': None}

if __name__ == '__main__':
    # Initialize database on first run
    init_db()
    
    print("=== Interview System Started ===")
    print("Admin Login: admin@gmail.com / admin123")
    print("Student Login: student@gmail.com / student123")
    print("===============================")
    
    app.run(debug=True, port=5003)