# ✅ app.py
from flask import Flask, render_template, request, redirect, session
import psycopg2
import subprocess
import os

app = Flask(__name__)
app.secret_key = 'secret123'

# PostgreSQL connection
conn = psycopg2.connect(
    dbname='behavioural',
    user='postgres',
    password='shivanirao1710',
    host='localhost',
    port='5432'
)
cur = conn.cursor()


# -------------------------
# Register New User
# -------------------------
@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']

        cur.execute("SELECT * FROM users WHERE username = %s", (username,))
        if cur.fetchone():
            return "Username already exists. Please try another."

        cur.execute("INSERT INTO users (username, password) VALUES (%s, %s)", (username, password))
        conn.commit()
        return redirect('/login')

    return render_template('register.html')

# -------------------------
# Login Page
# -------------------------
@app.route('/')
@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']

        cur.execute("SELECT * FROM users WHERE username = %s AND password = %s", (username, password))
        user = cur.fetchone()

        if user:
            session['username'] = username
            return redirect('/dashboard')
        else:
            return "Invalid username or password."

    return render_template('login.html')

# -------------------------
# Dashboard
# -------------------------
@app.route('/dashboard')
def dashboard():
    if 'username' not in session:
        return redirect('/')
    return render_template('dashboard.html', username=session['username'])

# -------------------------
# Start Interview (Run Python Script)
# -------------------------
@app.route('/start-interview', methods=['POST'])
def start_interview():
    username = session['username']
    env = os.environ.copy()
    env['LOGGED_IN_USER'] = username
    subprocess.run(["python", "interview/live_interview.py"], env=env)
    return redirect('/results')

# -------------------------
# Results Page
# -------------------------
@app.route('/results')
def results():
    if 'username' not in session:
        return redirect('/')

    username = session['username']

    cur.execute("SELECT timestamp, emotion FROM emotion_logs WHERE username = %s ORDER BY id", (username,))
    emotions = cur.fetchall()

    cur.execute("SELECT question, answer FROM interview_answers WHERE username = %s ORDER BY id", (username,))
    answers = cur.fetchall()

    return render_template('results.html',
                           emotions=[{'timestamp': e[0], 'emotion': e[1]} for e in emotions],
                           answers=[{'question': a[0], 'answer': a[1]} for a in answers])

# -------------------------
# Generate Report (LLM)
# -------------------------
@app.route('/generate-report', methods=['POST'])
def generate_report():
    if 'username' not in session:
        return redirect('/')

    username = session['username']
    env = os.environ.copy()
    env['LOGGED_IN_USER'] = username

    subprocess.run(["python", "generate_report.py"], env=env)
    return redirect('/view-report')
# -------------------------
# View Soft Skill Report
# -------------------------
@app.route('/view-report')
def view_report():
    if 'username' not in session:
        return redirect('/')

    username = session['username']

    cur.execute("SELECT question, answer, analysis FROM soft_skill_analysis WHERE username = %s", (username,))
    report = cur.fetchall()

    return render_template('report.html', report=[
        {'question': r[0], 'answer': r[1], 'analysis': r[2]} for r in report
    ])

# -------------------------
# Logout
# -------------------------
@app.route('/logout')
def logout():
    session.pop('username', None)
    return redirect('/')

# -------------------------
# Run App
# -------------------------
if __name__ == '__main__':
    app.run(debug=True)