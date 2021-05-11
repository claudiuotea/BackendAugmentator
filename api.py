import uuid
from flask import Flask, jsonify, request, make_response
from flask_cors import CORS, cross_origin
from flask_sqlalchemy import SQLAlchemy
from sqlalchemy import or_
from werkzeug.security import generate_password_hash, check_password_hash
import jwt
import datetime
from functools import wraps
from flask_mail import Mail, Message

import utils

app = Flask(__name__)
cors = CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'

# O sa utilizam secret key-ul pentru JWT
app.config['SECRET_KEY'] = 'JDMpower'

# configurari pentru email
app.config['MAIL_SERVER'] = 'smtp.gmail.com'
app.config['MAIL_PORT'] = 465
app.config['MAIL_USERNAME'] = 'claudiuoteaogc1@gmail.com'
app.config['MAIL_PASSWORD'] = 'veverita1999'
app.config['MAIL_USE_TLS'] = False
app.config['MAIL_USE_SSL'] = True

# aici salvam database-ul
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///E:/Anul 3/RESTapi/augmdbfinal.db'

db = SQLAlchemy(app)
# instanta pentru clasa care trimite mailuri
mail = Mail(app)


class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    # folosim asta pentru JWT, ca sa nu dam id-ul real in caz ca cineva decodeaza JWT-ul
    public_id = db.Column(db.String(50), unique=True)
    email = db.Column(db.String(50))
    username = db.Column(db.String(50))
    password = db.Column(db.String(50))
    admin = db.Column(db.Boolean)
    verified = db.Column(db.Boolean)


class ResetTokens(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    public_id = db.Column(db.String(50))
    token = db.Column(db.String(50))
    exp_date = db.Column(db.DateTime)


class VerifyTokens(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    public_id = db.Column(db.String(50))
    token = db.Column(db.String(50))


# vom folosi acest decorator peste toate call-urile care necesita autorizare
def token_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        token = None

        # verificam daca exista token-ul in header
        if 'x-access-token' in request.headers:
            token = request.headers['x-access-token']

        # returnam 401 daca nu
        if not token:
            return jsonify({'message': 'Token is missing!'}), 401

        # Incercam sa extragem din JWT public id-ul (care este unic) si sa gasim user-ul apartinator
        try:
            data = jwt.decode(token, app.config['SECRET_KEY'])
            current_user = User.query.filter_by(public_id=data['public_id']).first()
        except:
            # daca nu se poate, token-ul este invalid
            return jsonify({'message': 'Token invalid!'}), 401

        return f(current_user, *args, **kwargs)

    return decorated


@app.route("/user", methods=['GET'])
@token_required
@cross_origin()
def get_all_users(current_user):
    # doar admin poate vedea userii
    if not current_user.admin:
        return jsonify({'message': 'Permission denied!'}), 401

    # luam din database toate inregistrarile de useri
    users = User.query.all()
    output = []

    for user in users:
        user_data = {}
        user_data['public_id'] = user.public_id
        user_data['email'] = user.email
        user_data['username'] = user.username
        user_data['password'] = user.password
        user_data['admin'] = user.admin
        output.append(user_data)

    return jsonify({'users': output})


@app.route('/user/<public_id>', methods=['GET'])
@token_required
@cross_origin()
def get_one_user(current_user, public_id):
    # doar admin poate
    if not current_user.admin:
        return jsonify({'message': 'Permission denied!'}), 401

    # il cautam dupa user_id
    user = User.query.filter_by(public_id=public_id).first()
    # daca nu gasim un user cu acel public id
    if not user:
        return jsonify({'message': 'No user found!'})

    user_data = {}
    user_data['public_id'] = user.public_id
    user_data['email'] = user.email
    user_data['username'] = user.username
    user_data['password'] = user.password
    user_data['admin'] = user.admin

    return jsonify({'user': user_data})


@app.route('/register', methods=['POST'])
@cross_origin()
def register_user():
    # salvam datele din request
    data = request.get_json()

    #verificam daca user-ul exista deja in database
    user = User.query.filter(or_(User.username==data['username'],User.email==data['email'])).all()

    if user:
        return jsonify({'message': 'User already registered!'}), 403

    # hash-uim parola
    hashed_password = generate_password_hash(data['password'], method="sha256")
    # creez un user nou
    new_user = User(public_id=str(uuid.uuid4()), email=data['email'], username=data['username']
                    , password=hashed_password, admin=False, verified=False)
    # il adaugam in database
    db.session.add(new_user)

    # creez si trimit link-ul pt verificare cont
    url, token = utils.Utils.store_verify_token(new_user.public_id)

    insertToken = VerifyTokens(token=token,public_id=new_user.public_id)
    db.session.add(insertToken)

    # ii trimit mail cu link-ul
    msg = Message('Verify your account', sender=app.config['MAIL_USERNAME'], recipients=[new_user.email])
    msg.body = "Please access this link to verify your account: " + url + " . If you do not verify your account you won't be able to use our features."
    mail.send(msg)
    db.session.commit()
    return jsonify({'message:': 'New user created!'}), 200


@app.route('/verifyaccount', methods=["POST"])
def verifyAccount():
    # preluam datele
    data = request.get_json()

    # gasim token-ul in baza de date si vedem daca este al user-ului
    token = VerifyTokens.query.filter_by(token=data['token']).first()

    # daca nu am gasit token-ul sau nu este al user-ului atunci nu avem voie sa schimbam parola
    if not token or token.public_id != data['public_id']:
        return jsonify({'message': 'Not authorized!'}), 403

    # setam contul ca si verificat si stergem token-ul
    user = User.query.filter_by(public_id=data['public_id']).first()
    user.verified = True
    db.session.delete(token)
    db.session.commit()
    return jsonify({'message': 'Success!'}), 200


@app.route('/user/<public_id>', methods=['PUT'])
@token_required
@cross_origin()
def promote_user(current_user, public_id):
    # doar admin poate vedea userii
    if not current_user.admin:
        return jsonify({'message': 'Permission denied!'}), 401

    # il cautam dupa user_id
    user = User.query.filter_by(public_id=public_id).first()
    # daca nu gasim un user cu acel public id
    if not user:
        return jsonify({'message': 'No user found!'})
    # ii dam drepturi de admin
    user.admin = True
    db.session.commit()

    return jsonify({'message:': 'The user is now an admin!'})


@app.route('/user/<public_id>', methods=['DELETE'])
@token_required
@cross_origin()
def delete_user(current_user, public_id):
    # doar admin poate vedea userii
    if not current_user.admin:
        return jsonify({'message': 'Permission denied!'}), 401

    # il cautam dupa user_id
    user = User.query.filter_by(public_id=public_id).first()
    # daca nu gasim un user cu acel public id
    if not user:
        return jsonify({'message': 'No user found!'})

    # stergem user-ul
    db.session.delete(user)
    db.session.commit()

    return jsonify({'message': 'The user has been deleted!'})


@app.route('/login')
@cross_origin()
def login():
    # preluam informatiile pt autorizare
    auth = request.authorization

    # daca nu exista sau nu sunt complete, returnam 401
    if not auth or not auth.username or not auth.password:
        return make_response('Could not verify', 401, {'WWW-Authenticate': 'Basic realm="Login required"'})

    # preluam user-ul din database dupa username
    user = User.query.filter_by(username=auth.username).first()

    # daca nu e gasit anuntam ca nu exista
    if not user:
        return make_response('Could not verify', 401, {'WWW-Authenticate': 'Basic realm="Login required"'})

    # daca exista verificam ca parola e corecta
    if check_password_hash(user.password, auth.password):
        #daca contul nu este verificat, oprim login-ul
        if user.verified == False:
            return jsonify({'message': 'Please verify account!'}), 403

        # generam token-ul ptc parola e corecta
        token = jwt.encode(
            {'public_id': user.public_id, 'exp': datetime.datetime.utcnow() + datetime.timedelta(hours=24)}
            , app.config['SECRET_KEY'])
        # il returnam
        return jsonify({'token': token.decode('UTF-8')})

    # parola nu e corecta, returnam 401
    return make_response('Could not verify', 401, {'WWW-Authenticate': 'Basic realm="Login required"'})


@app.route('/forgotpass', methods=["POST"])
def forgotPass():
    # preluam informatia din request
    data = request.get_json()

    # il cautam dupa email sa vedem daca exista
    user = User.query.filter_by(email=data['email']).first()

    if not user:
        return jsonify({'message': 'Permission denied!'}), 404

    # generez link-ul pentru resetare parola
    url, token, exp_date = utils.Utils.store_reset_token(user.public_id)

    # salvez in database token-ul si data de expirare
    reset_token = ResetTokens(public_id=user.public_id, token=token, exp_date=exp_date)
    db.session.add(reset_token)
    db.session.commit()

    # ii trimit mail cu link-ul
    msg = Message('Reset your password', sender=app.config['MAIL_USERNAME'], recipients=[user.email])
    msg.body = "Please access this link to reset your password within the next 6 hours: " + url
    mail.send(msg)
    return jsonify({'message': 'Email sent!'}), 200


@app.route('/resetpass', methods=["POST"])
def resetPass():
    # preluam datele
    data = request.get_json()

    # gasim token-ul in baza de date si vedem daca este al user-ului
    token = ResetTokens.query.filter_by(token=data['token']).first()

    # daca nu am gasit token-ul sau nu este al user-ului atunci nu avem voie sa schimbam parola
    if not token or token.public_id != data['public_id']:
        return jsonify({'message': 'Not authorized!'}), 403

    #daca am gasit token atunci verificam daca sunt mai multe ca sa le stergem pe toate
    tokens = ResetTokens.query.filter_by(public_id=data['public_id']).all()


    # daca am gasit token-ul si este al user-ului, verificam daca inca este valid token-ul
    if token.exp_date > datetime.datetime.now():
        # resetam parola si stergem token-ul
        user = User.query.filter_by(public_id=data['public_id']).first()
        user.password = generate_password_hash(data['password'], method="sha256")

        #sterg toate token-urile user-ului
        for t in tokens:
            db.session.delete(t)
        db.session.commit()
        return jsonify({'message': 'Success!'}), 200
    else:
        #si daca sunt expirate le sterg pe toate
        for t in tokens:
            db.session.delete(t)
        db.session.commit()
        return jsonify({'message': 'Token expired!'}), 403


if __name__ == '__main__':
    app.run(debug=True)
