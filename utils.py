import datetime
import secrets
from zipfile import ZipFile

class Utils:
    # forma de baza url-ului
    BASE_URL = "http://localhost:3000"

    '''O sa fie folosita pentru a crea un link securizat de reset password'''

    @staticmethod
    def create_random_token():
        return secrets.token_urlsafe(16)

    # Salveaza automat in database un token, un expiration date, si userul al cui e token ul
    @staticmethod
    def store_reset_token(public_id):
        # creaza token
        token = Utils.create_random_token()
        # data de expirare a token-ului este peste 6 ore
        exp_date = datetime.datetime.now() + datetime.timedelta(hours=6)

        # construiesc link-ul pe care il pasez
        url = Utils.BASE_URL + "/account/resetpass?" + "id=" + public_id + "&token=" + token
        return url, token, exp_date

    # Salveaza automat in database un token pentru verificarea contului, si userul al cui e token ul
    @staticmethod
    def store_verify_token(public_id):
        # creaza token
        token = Utils.create_random_token()

        # construiesc link-ul pe care il pasez
        url = Utils.BASE_URL +"/account/verifyaccount?" + "id=" + public_id + "&token=" + token
        return url, token

    @staticmethod
    def unzip_file(path_to_zip,path_to_unzip):
        with ZipFile(path_to_zip, 'r') as zip_ref:
            zip_ref.extractall(zip_ref)
