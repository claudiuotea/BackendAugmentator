import uuid
from zipfile import ZipFile
from flask import Flask, jsonify, request, make_response, send_from_directory, redirect
from flask_cors import CORS, cross_origin
from flask_jwt_extended import JWTManager, create_access_token, jwt_required, get_jwt_identity, create_refresh_token,current_user
from flask_sqlalchemy import SQLAlchemy
from sqlalchemy import or_
from werkzeug.security import generate_password_hash, check_password_hash
import datetime
from flask_mail import Mail, Message
import utils
from os import path
import os
from Augmentator import Augmentator

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

#configurari JWT
app.config["JWT_ACCESS_TOKEN_EXPIRES"] = datetime.timedelta(hours=6)
app.config["JWT_REFRESH_TOKEN_EXPIRES"] = datetime.timedelta(days=30)
#path-ul unde userii salveaza datele
app.config["USERS_FOLDER"] = "users"
app.config["DOWNLOAD_FOLDER"] ="users/AUGMENTED"
# aici salvam database-ul
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///E:\Anul 3\LICENTA\RESTapi/augmdbfinal.db'
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0


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

#aici vor fi salvate date pentru fiecare augmentare facuta de catre un user
#de aici se vor crea statistici mai tarziu
class Augmentation(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.String(50))
    isClahe = db.Column(db.String(50))
    isGray = db.Column(db.String(50))
    isFlip = db.Column(db.String(50))
    isFlipBase = db.Column(db.String(50))
    isFlipClahe = db.Column(db.String(50))
    isFlipGray = db.Column(db.String(50))
    isErase = db.Column(db.String(50))
    isEraseBase = db.Column(db.String(50))
    isEraseGray = db.Column(db.String(50))
    isEraseClahe = db.Column(db.String(50))
    link = db.Column(db.String(50))
    filename = db.Column(db.String(50))


@app.before_request
def consume_request_body():
    """ Consumes the request body before handling a request to fix uwsgi+nginx problems
    See https://github.com/vimalloc/flask-jwt-extended/issues/253#issuecomment-505222118
    for more details """
    request.data


#returneaza toate augmentarile facute de user-ul curent
@app.route("/augmentations", methods=["GET"])
@jwt_required()
def getUserAugmentations():
    #daca e admin va primi datele de la toti userii
    if current_user.admin == True:
        augmentations = Augmentation.query.all()
    #daca nu e admin va primi doar datele lui
    else:
        augmentations = Augmentation.query.filter_by(user_id=current_user.public_id).all()

    #acum numar cate din augmentari au fost folosite
    output = {}
    links = []
    filenames = []

    clahe, gray, flip, erase, flipBase, flipClahe, flipGray, eraseBase, eraseClahe, eraseGray = 0,0,0,0,0,0,0,0,0,0
    for aug in augmentations:
        if aug.isClahe == "true":
            clahe += 1
        if aug.isGray == "true":
            gray += 1
        if aug.isFlipBase == "true":
            flip += 1
            flipBase += 1
        if aug.isFlipGray == "true":
            flip += 1
            flipGray += 1
        if aug.isFlipClahe == "true":
            flip += 1
            flipClahe += 1
        if aug.isEraseBase == "true":
            erase += 1
            eraseBase += 1
        if aug.isEraseGray == "true":
            erase += 1
            eraseGray += 1
        if aug.isEraseClahe == "true":
            erase += 1
            eraseClahe += 1
        links.append(aug.link)
        filenames.append(aug.filename)

    output["clahe"] = clahe
    output["gray"] = gray
    output["flip"] = flip
    output["erase"] = erase
    output["flipBase"] = flipBase
    output["flipClahe"] = flipClahe
    output["flipGray"] = flipGray
    output["eraseGray"] = eraseGray
    output["eraseBase"] = eraseBase
    output["eraseClahe"] = eraseClahe
    output["links"] = links
    output["filenames"] = filenames
    return jsonify({'augmentations': output})

#O sa fie cu /userpath pentru ca aici va fi public id-ul (care e unic,deci link-ul va fi unic)
@app.route("/download/<publicId>/<fileName>", methods=["GET"])
@jwt_required()
def download(publicId,fileName):
    #creez path-ul pentru download
    folderPath = os.getcwd()+"/"+app.config["DOWNLOAD_FOLDER"] + "/" + publicId
    #verific daca user-ul curent e cel care are public id de unde vrea sa descarce, altfel e unauthorized
    if current_user.public_id != publicId:
        return make_response("You are not authorized to access this link", 403)

    response = send_from_directory(folderPath,fileName,as_attachment=True, cache_timeout = 0)
    response.headers.add("Access-Control-Allow-Origin", "*")
    return response


@app.route("/uploadfile",methods=["POST"])
@jwt_required()
def uploadFileFromClient():
        file = request.files.get("myFile")

        #path-ul user-ului
        savePath = os.getcwd() +"\\" + app.config["USERS_FOLDER"] + "\\" + current_user.public_id
        cleanPath = savePath + "\\" + "CLEAN"
        #path-ul unde salveaza arhiva augmentata
        saveArchivePath = os.getcwd() + "\\" +app.config["USERS_FOLDER"] + "\\AUGMENTED\\" + current_user.public_id

        #daca exista deja un fisier cu numele asta cer sa-l schimbe
        if os.path.exists(saveArchivePath + "\\" +file.filename):
            return jsonify("Please select a different name for your archive, not one you already used"),403

        #deschide fisierul cu ZipFile ca sa-l dezarhivezi direct din request
        file_like_object = file.stream._file
        zipfile_ob = ZipFile(file_like_object)
        zipfile_ob.extractall(cleanPath)

        #creez parametrii pentru augmentare
        augmData = request.form

        clahe,grayscale,flip,erase,rotate,flipOption,eraseOption = Augmentator.convertParams(
            augmData["isClahe"],augmData["isGray"],augmData["isFlip"],augmData["isErase"],
            augmData["isFlipBase"],augmData["isFlipClahe"],augmData["isFlipGray"],
            augmData["isEraseBase"],augmData["isEraseClahe"],augmData["isEraseGray"],
            augmData["flipProbability"],augmData["eraseProbability"],augmData["rotateProbability"]
        )
        augmentator = Augmentator(clahe=clahe,grayscale=grayscale,flip=flip,erase=erase
                                  ,rotate=rotate,flipOption=flipOption,eraseOption=eraseOption
                                  ,datasetPath=savePath, archiveName=file.filename,saveArchivePath=saveArchivePath)

        augmentator.applyAugmentations()

        #dupa augmentare il redirectionez la download automat [un link care contine
        #in url parameters datele pe care el le va trimite la server automat pentru download
        linkToDownload = utils.Utils.store_download_link(current_user.public_id, file.filename)

        # salvam in database parametrii augmentarii
        augm = Augmentation(user_id=current_user.public_id, isClahe=augmData["isClahe"], isGray=augmData["isGray"],
                            isFlip=augmData["isFlip"],
                            isErase=augmData["isErase"], isFlipBase=augmData["isFlipBase"],
                            isFlipGray=augmData["isFlipGray"], isFlipClahe=augmData["isFlipClahe"],
                            isEraseBase=augmData["isEraseBase"], isEraseClahe=augmData["isEraseClahe"],
                            isEraseGray=augmData["isEraseGray"], link=linkToDownload, filename=file.filename)
        db.session.add(augm)
        db.session.commit()

        resp = make_response(linkToDownload,200)
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
        return make_response('Wrong email or password', 401, {'WWW-Authenticate': 'Basic realm="Login required"'})

    # preluam user-ul din database dupa username
    user = User.query.filter_by(username=auth.username).first()

    # daca nu e gasit anuntam ca nu exista
    if not user:
        return make_response('Could not verify', 401, {'WWW-Authenticate': 'Basic realm="Login required"'})

    # daca exista verificam ca parola e corecta
    if check_password_hash(user.password, auth.password):
        # daca contul nu este verificat, oprim login-ul
        if user.verified == False:
            return make_response('Please verify your account!', 403)

        access_token = create_access_token(user.public_id)
        refresh_token = create_refresh_token(user.public_id)
        return jsonify({'AccessToken': access_token,'RefreshToken':refresh_token})

    # parola nu e corecta, returnam 401
    return make_response('Wrong credentials!', 401, {'WWW-Authenticate': 'Basic realm="Login required"'})


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
        os.mkdir(currentPath + "\\" + app.config["USERS_FOLDER"])

    userPath = os.getcwd() + "\\" + app.config["USERS_FOLDER"] + "\\" + new_user.public_id
    os.mkdir(userPath)
    #se creeaza si un folder unde vor fi storate toate datele augmentate, daca nu exista
    #aici fiecare user va avea din nou un folder propriu unde vor fi storate datele augmentate
    if not path.exists(app.config["USERS_FOLDER"]+"\\AUGMENTED"):
        currentPath = os.getcwd()
        os.mkdir(currentPath + "\\" + app.config["USERS_FOLDER"]+"\\AUGMENTED")
    augmUserPath = os.getcwd() + "\\" + app.config["USERS_FOLDER"] + "\\AUGMENTED\\" + new_user.public_id
    os.mkdir(augmUserPath)

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
        return jsonify({'message': 'User not found!'}), 403

    # generez link-ul pentru resetare parola
    url, token, exp_date = utils.Utils.store_reset_token(user.public_id)

    # salvez in database token-ul si data de expirare
    reset_token = ResetTokens(public_id=user.public_id, token=token, exp_date=exp_date)
    db.session.add(reset_token)
    db.session.commit()

    # ii trimit mail cu link-ul
    msg = Message('Reset your password', sender=app.config['MAIL_USERNAME'], recipients=[user.email])
    msg.body = "Please access this link to reset your password as soon as possible: " + url
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
