import uuid
from zipfile import ZipFile

import flask_jwt_extended
from flask import Flask, jsonify, request, make_response
from flask_cors import CORS, cross_origin
from flask_jwt_extended import JWTManager, create_access_token, jwt_required, get_jwt_identity, create_refresh_token,current_user
from flask_jwt_extended.default_callbacks import default_expired_token_callback, default_invalid_token_callback, \
    default_unauthorized_callback, default_revoked_token_callback
from flask_sqlalchemy import SQLAlchemy
from sqlalchemy import or_
from werkzeug.security import generate_password_hash, check_password_hash
import datetime
from flask_mail import Mail, Message
import utils
from os import path
import os

app = Flask(__name__)
# cors = CORS(app)
CORS(app)

# O sa utilizam secret key-ul pentru JWT
app.config["JWT_SECRET_KEY"] = 'JDMpower'
# jwt
jwt = JWTManager(app)

app.config['CORS_HEADERS'] = 'Content-Type'

# configurari pentru email
app.config['MAIL_SERVER'] = 'smtp.gmail.com'
app.config['MAIL_PORT'] = 465
app.config['MAIL_USERNAME'] = 'claudiuoteaogc1@gmail.com'
app.config['MAIL_PASSWORD'] = 'veverita1999'
app.config['MAIL_USE_TLS'] = False
app.config['MAIL_USE_SSL'] = True
app.config["JWT_ACCESS_TOKEN_EXPIRES"] = datetime.timedelta(seconds=15)
app.config["JWT_REFRESH_TOKEN_EXPIRES"] = datetime.timedelta(days=30)
app.config["USERS_FOLDER"] = "users"
#500mb maximum
app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024
# aici salvam database-ul
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///E:/Anul 3/RESTapi/augmdbfinal.db'

db = SQLAlchemy(app)
# instanta pentru clasa care trimite mailuri
mail = Mail(app)

@app.before_request
def consume_request_body():
    """ Consumes the request body before handling a request to fix uwsgi+nginx problems
    See https://github.com/vimalloc/flask-jwt-extended/issues/253#issuecomment-505222118
    for more details """
    request.data

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


@app.route("/uploadfile",methods=["POST"])
#@jwt_required()
def uploadFileFromClient():
    file = request.files.get("myFile")
    #path-ul user-ului
    # savePath = os.getcwd() +"\\" + app.config["USERS_FOLDER"] + "\\" + current_user.public_id
    savePath = os.getcwd() +"\\" + app.config["USERS_FOLDER"] + "\\" + current_user.public_id
    # file_like_object = file.stream._file
    # zipfile_ob = ZipFile(file_like_object)
    # zipfile_ob.extractall(savePath)


    #data.save(os.path.join(savePath,data.filename))
    resp = make_response('Success!', 200)
    # Enable Access-Control-Allow-Origin
    #resp.headers.add("Access-Control-Allow-Origin", "*")
    return resp
# We are using the `refresh=True` options in jwt_required to only allow
# refresh tokens to access this route.
@app.route("/refresh", methods=["POST"])
@jwt_required(refresh=True)
def refresh():
    identity = get_jwt_identity()
    access_token = create_access_token(identity=identity)
    print(access_token)
    return jsonify(access_token=access_token)


@jwt.user_identity_loader
def user_identity_lookup(user):
    return user


@jwt.user_lookup_loader
def user_lookup_callback(jwt_header, jwt_data):
    identity = jwt_data["sub"]
    return User.query.filter_by(public_id=identity).one_or_none()




@app.route("/contact", methods=["POST"])
def sendEmail():
    data = request.get_json()["mail"]

    # ii trimit mail cu link-ul
    print(data["subject"])
    msg = Message(data['subject'], sender=data['mail'], recipients=[app.config['MAIL_USERNAME']])
    msg.body = "Send from: "+data['mail']+"\n" + data['message'] + "\nPhone number: " + data['phoneNumber']
    mail.send(msg)
    return make_response('Success!', 200)
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
        # daca contul nu este verificat, oprim login-ul
        if user.verified == False:
            return jsonify({'message': 'Please verify account!'}), 403

        access_token = create_access_token(user.public_id)
        refresh_token = create_refresh_token(user.public_id)
        return jsonify({'AccessToken': access_token,'RefreshToken':refresh_token})

    # parola nu e corecta, returnam 401
    return make_response('Could not verify', 401, {'WWW-Authenticate': 'Basic realm="Login required"'})


@app.route("/user", methods=['GET'])
@jwt_required()
@cross_origin()
def get_all_users():
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
        user_data['isVerified'] = user.verified
        user_data['isAdmin'] = user.admin
        output.append(user_data)

    return jsonify({'users': output})


@app.route('/user/<public_id>', methods=['GET'])
@jwt_required()
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

    # verificam daca user-ul exista deja in database
    user = User.query.filter(or_(User.username == data['username'], User.email == data['email'])).all()

    if user:
        return jsonify({'message': 'User already registered!'}), 403

    # hash-uim parola
    hashed_password = generate_password_hash(data['password'], method="sha256")
    # creez un user nou
    new_user = User(public_id=str(uuid.uuid4()), email=data['email'], username=data['username']
                    , password=hashed_password, admin=False, verified=False)
    # il adaugam in database
    db.session.add(new_user)

    # pentru fiecare user creez un folder pe server, unde-si storeaza datele
    # verific prima data daca exista deja un folder de baza pentru toti userii, altfel il creez
    if not path.exists(app.config["USERS_FOLDER"]):
        currentPath = os.getcwd()
        os.mkdir(currentPath + "\\" +app.config["USERS_FOLDER"])

    userPath = os.getcwd() +"\\" + app.config["USERS_FOLDER"] + "\\" + new_user.public_id
    os.mkdir(userPath)

    # creez si trimit link-ul pt verificare cont
    url, token = utils.Utils.store_verify_token(new_user.public_id)

    insertToken = VerifyTokens(token=token, public_id=new_user.public_id)
    db.session.add(insertToken)

    # ii trimit mail cu link-ul
    msg = Message('Verify your account', sender=app.config['MAIL_USERNAME'], recipients=[new_user.email])
    msg.body = "Please access this link to verify your account: " + url + " . If you do not verify your account you won't be able to use our features."
    mail.send(msg)
    db.session.commit()
    return jsonify({'message:': 'New user created!'}), 200


@cross_origin()
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

@cross_origin()
@app.route('/verifybyadmin', methods=["PUT"])
@jwt_required()
def verifyByAdmin():
    # preluam datele
    data = request.get_json()

    if not current_user.admin:
        return jsonify({'message': 'Permission denied!'}), 401
    # setam contul ca si verificat si stergem token-ul
    user = User.query.filter_by(public_id=data['public_id']).first()
    user.verified = True
    db.session.commit()
    return jsonify({'message': 'Success!'}), 200

@app.route('/user/<public_id>', methods=['PUT'])
@jwt_required()
@cross_origin()
def promote_user(public_id):
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
@jwt_required()
@cross_origin()
def delete_user(public_id):
    # doar admin poate sterge userii
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


@app.route('/forgotpass', methods=["POST"])
@cross_origin()
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
@cross_origin()
def resetPass():
    # preluam datele
    data = request.get_json()

    # gasim token-ul in baza de date si vedem daca este al user-ului
    token = ResetTokens.query.filter_by(token=data['token']).first()

    # daca nu am gasit token-ul sau nu este al user-ului atunci nu avem voie sa schimbam parola
    if not token or token.public_id != data['public_id']:
        return jsonify({'message': 'Not authorized!'}), 403

    # daca am gasit token atunci verificam daca sunt mai multe ca sa le stergem pe toate
    tokens = ResetTokens.query.filter_by(public_id=data['public_id']).all()

    # daca am gasit token-ul si este al user-ului, verificam daca inca este valid token-ul
    if token.exp_date > datetime.datetime.now():
        # resetam parola si stergem token-ul
        user = User.query.filter_by(public_id=data['public_id']).first()
        user.password = generate_password_hash(data['password'], method="sha256")

        # sterg toate token-urile user-ului
        for t in tokens:
            db.session.delete(t)
        db.session.commit()
        return jsonify({'message': 'Success!'}), 200
    else:
        # si daca sunt expirate le sterg pe toate
        for t in tokens:
            db.session.delete(t)
        db.session.commit()
        return jsonify({'message': 'Token expired!'}), 403


@app.route('/checkadmin', methods=["GET"])
@jwt_required()
@cross_origin()
def checkAdmin():
    # returnez daca e admin sau nu
    return jsonify({'admin': current_user.admin})


if __name__ == '__main__':
    app.run(debug=True)
