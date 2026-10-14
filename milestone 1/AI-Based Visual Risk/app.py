"""Visual Risk Flask application."""

from datetime import date, datetime
import html
import os

from dotenv import load_dotenv
from flask import Flask, flash, redirect, render_template, render_template_string, request, url_for
from flask_login import LoginManager, UserMixin, current_user, login_required, login_user, logout_user
from flask_sqlalchemy import SQLAlchemy
from jinja2 import FileSystemLoader
from werkzeug.security import check_password_hash, generate_password_hash


class UnescapingTemplateLoader(FileSystemLoader):
    """Loads the existing templates after converting HTML entities back to text."""

    def get_source(self, environment, template):
        source, filename, uptodate = super().get_source(environment, template)
        return html.unescape(source), filename, uptodate


load_dotenv()
app = Flask(__name__, template_folder="templets")
app.jinja_loader = UnescapingTemplateLoader(os.path.join(app.root_path, "templets"))
app.config["SECRET_KEY"] = os.getenv("SECRET_KEY", "dev-key-change-in-production")
app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite:///digital_twin.db"
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False
db = SQLAlchemy(app)

login_manager = LoginManager(app)
login_manager.login_view = "login"


class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password_hash = db.Column(db.String(255), nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    is_admin = db.Column(db.Boolean, default=False)

    def set_password(self, password):
        self.password_hash = generate_password_hash(password)

    def check_password(self, password):
        return check_password_hash(self.password_hash, password)


class Transaction(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey("user.id"), nullable=False)
    date = db.Column(db.Date, nullable=False, default=date.today)
    category = db.Column(db.String(50), nullable=False)
    amount = db.Column(db.Float, nullable=False)
    type = db.Column(db.String(10), nullable=False)


class StudyLog(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey("user.id"), nullable=False)
    date = db.Column(db.Date, nullable=False, default=date.today)
    hours = db.Column(db.Float, nullable=False)
    subject = db.Column(db.String(100), nullable=False)
    productivity_score = db.Column(db.Integer, nullable=False)


class FitnessLog(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey("user.id"), nullable=False)
    date = db.Column(db.Date, nullable=False, default=date.today)
    activity = db.Column(db.String(50), nullable=False)
    duration_min = db.Column(db.Integer, nullable=False)
    calories_burned = db.Column(db.Integer, default=0)


@login_manager.user_loader
def load_user(user_id):
    return db.session.get(User, int(user_id))


SIMPLE_PAGE = """
<!doctype html><title>{{ title }} · Visual Risk</title>
<main style="font-family:Arial;max-width:760px;margin:3rem auto;padding:0 1rem">
  <p><a href="{{ url_for('index') }}">← Home</a></p>
  {% for message in get_flashed_messages() %}<p style="background:#fff3cd;padding:.7rem">{{ message }}</p>{% endfor %}
  <h1>{{ title }}</h1>{{ body|safe }}
</main>
"""


def simple_page(title, body):
    return render_template_string(SIMPLE_PAGE, title=title, body=body)


@app.route("/")
def index():
    # This deliberately retains the existing index.html -> base.html relationship.
    return render_template("index.html")


@app.route("/register", methods=["GET", "POST"])
def register():
    # return render_template("register.html")
    if request.method == 'POST':
        username = request.form.get('username', '').strip()
        email = request.form.get('email', '').strip()
        password = request.form.get('password', '')
        confirm_password = request.form.get('confirm_password', '')
        if not username or not email or not password:
            flash('All fields are required.')
            return redirect(url_for('register'))
        if password != confirm_password:
            flash('Passwords do not match.')
            return redirect(url_for('register'))
        if len(password) < 8:
            flash('Password must be at least 8 characters.')
            return redirect(url_for('register'))
        if User.query.filter_by(username=username).first():
            flash('Username already exists.')
            return redirect(url_for('register'))
        if User.query.filter_by(email=email).first():
            flash('Email already registered.')
            return redirect(url_for('register'))
        user = User(username=username, email=email)
        user.set_password(password)
        # First user becomes admin – but now we already have a default admin, so this is optional
        # if User.query.count() == 0:
        #     user.is_admin = True
        db.session.add(user)
        db.session.commit()
        flash('Registration successful! Please log in.')
        return redirect(url_for('login'))
    return render_template('register.html')

@app.route("/login", methods=["GET", "POST"])
def login():
    if current_user.is_authenticated:
        return redirect(url_for("user_dashboard"))
    if request.method == "POST":
        user = User.query.filter_by(username=request.form.get("username", "").strip()).first()
        if user and user.check_password(request.form.get("password", "")):
            login_user(user)
            return redirect(url_for("user_dashboard"))
        flash("Invalid username or password.")
    return render_template("login.html")


@app.route("/logout")
@login_required
def logout():
    logout_user()
    return redirect(url_for("index"))


@app.route("/user/dashboard")
@login_required
def user_dashboard():
    return simple_page("Dashboard", f"<p>Welcome, {current_user.username}.</p><p>Use the navigation menu to track your finances, study, and fitness.</p>")


@app.route("/user/financial")
@login_required
def user_financial():
    return simple_page("Financial tracker", "<p>Your financial tracker is ready for the restored page templates.</p>")


@app.route("/user/study")
@login_required
def user_study():
    return simple_page("Study tracker", "<p>Your study tracker is ready for the restored page templates.</p>")


@app.route("/user/fitness")
@login_required
def user_fitness():
    return simple_page("Fitness tracker", "<p>Your fitness tracker is ready for the restored page templates.</p>")


@app.route("/user/simulation")
@login_required
def user_simulation():
    return simple_page("Simulation", "<p>Simulation is available after its page template is restored.</p>")


@app.route("/user/assistant")
@login_required
def user_assistant():
    return simple_page("AI assistant", "<p>The AI assistant is available after its page template is restored.</p>")


@app.route("/settings")
@login_required
def settings():
    return simple_page("Settings", "<p>Settings page.</p>")


@app.route("/admin/dashboard")
@login_required
def admin_dashboard():
    return simple_page("Admin dashboard", "<p>Admin dashboard.</p>")


@app.route("/admin/users")
@login_required
def admin_users():
    return simple_page("Admin users", "<p>Admin users.</p>")


@app.route("/privacy")
def privacy():
    return simple_page("Privacy", "<p>Privacy information.</p>")


@app.route("/terms")
def terms():
    return simple_page("Terms", "<p>Terms of use.</p>")


@app.route("/contact")
def contact():
    return render_template("contact.html")


@app.route("/about")
def about():
    return render_template("about.html")


with app.app_context():
    db.create_all()


if __name__ == "__main__":
    app.run(debug=True)
