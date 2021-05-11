import uuid

from flask import Flask,jsonify,request,make_response
from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import generate_password_hash,check_password_hash
import jwt
import datetime
from functools import wraps


app = Flask(__name__)

#O sa utilizam secret key-ul pentru JWT
app.config['SECRET_KEY'] = 'JDMpower'
#aici salvam database-ul
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///E:/Anul 3/RESTapi/augm.db'

db = SQLAlchemy(app)

class User(db.Model):
    id = db.Column(db.Integer, primary_key = True)
    #folosim asta pentru JWT, ca sa nu dam id-ul real in caz ca cineva decodeaza JWT-ul
    public_id = db.Column(db.String(50), unique=True)
    email = db.Column(db.String(50))
    username = db.Column(db.String(50))
    password = db.Column(db.String(50))
    admin = db.Column(db.Boolean)


#vom folosi acest decorator peste toate call-urile care necesita autorizare
def token_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        token = None

        #verificam daca exista token-ul in header
        if 'x-access-token' in request.headers:
            token = request.headers['x-access-token']

        #returnam 401 daca nu
        if not token:
            return jsonify({'message' : 'Token is missing!'}) , 401

        #Incercam sa extragem din JWT public id-ul (care este unic) si sa gasim user-ul apartinator
        try:
            data = jwt.decode(token, app.config['SECRET_KEY'])
            current_user = User.query.filter_by(public_id=data['public_id']).first()
        except:
            #daca nu se poate, token-ul este invalid
            return jsonify({'message' : 'Token invalid!'}), 401

        return f(current_user, *args, **kwargs)
    return decorated


@app.route("/user", methods=['GET'])
@token_required
def get_all_users(current_user):
    #doar admin poate vedea userii
    if not current_user.admin:
        return jsonify({'message': 'Permission denied!'}), 401

    #luam din database toate inregistrarile de useri
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

    return jsonify({'users':output})

@app.route('/user/<public_id>', methods=['GET'])
@token_required
def get_one_user(current_user, public_id):
    # doar admin poate
    if not current_user.admin:
        return jsonify({'message': 'Permission denied!'}), 401

    #il cautam dupa user_id
    user = User.query.filter_by(public_id=public_id).first()
    #daca nu gasim un user cu acel public id
    if not user:
        return jsonify({'message':'No user found!'})

    user_data = {}
    user_data['public_id'] = user.public_id
    user_data['email'] = user.email
    user_data['username'] = user.username
    user_data['password'] = user.password
    user_data['admin'] = user.admin


    return jsonify({'user':user_data})

@app.route('/user', methods=['POST'])
def register_user():
    #salvam datele din request
    data = request.get_json()

    #hash-uim parola
    hashed_password = generate_password_hash(data['password'],method="sha256")
    #creez un user nou
    new_user = User(public_id=str(uuid.uuid4()), email = data['email'], username = data['username']
                    , password = hashed_password, admin = False)
    #il adaugam in database
    db.session.add(new_user)
    db.session.commit()
    return jsonify({'message:':'New user created!'})

@app.route('/user/<public_id>', methods=['PUT'])
@token_required
def promote_user(current_user, public_id):
    # doar admin poate vedea userii
    if not current_user.admin:
        return jsonify({'message': 'Permission denied!'}), 401

    # il cautam dupa user_id
    user = User.query.filter_by(public_id=public_id).first()
    # daca nu gasim un user cu acel public id
    if not user:
        return jsonify({'message': 'No user found!'})
    #ii dam drepturi de admin
    user.admin = True
    db.session.commit()

    return jsonify({'message:':'The user is now an admin!'})

@app.route('/user/<public_id>', methods=['DELETE'])
@token_required
def delete_user(current_user, public_id):
    # doar admin poate vedea userii
    if not current_user.admin:
        return jsonify({'message': 'Permission denied!'}), 401

    # il cautam dupa user_id
    user = User.query.filter_by(public_id=public_id).first()
    # daca nu gasim un user cu acel public id
    if not user:
        return jsonify({'message': 'No user found!'})

    #stergem user-ul
    db.session.delete(user)
    db.session.commit()

    return jsonify({'message': 'The user has been deleted!'})

@app.route('/login')
def login():
    #preluam informatiile pt autorizare
    auth = request.authorization

    #daca nu exista sau nu sunt complete, returnam 401
    if not auth or not auth.username or not auth.password:
        return make_response('Could not verify', 401, {'WWW-Authenticate' : 'Basic realm="Login required"'})

    #preluam user-ul din database dupa username
    user = User.query.filter_by(username=auth.username).first()

    #daca nu e gasit anuntam ca nu exista
    if not user:
        return make_response('Could not verify', 401, {'WWW-Authenticate' : 'Basic realm="Login required"'})

    #daca exista verificam ca parola e corecta
    if check_password_hash(user.password, auth.password):
        #generam token-ul ptc parola e corecta
        token = jwt.encode({'public_id': user.public_id, 'exp' : datetime.datetime.utcnow() + datetime.timedelta(hours=24)}
                           ,app.config['SECRET_KEY'])
        #il returnam
        return jsonify({'token' : token.decode('UTF-8')})

    #parola nu e corecta, returnam 401
    return make_response('Could not verify', 401, {'WWW-Authenticate': 'Basic realm="Login required"'})

if __name__ == '__main__':
    app.run(debug=True)

