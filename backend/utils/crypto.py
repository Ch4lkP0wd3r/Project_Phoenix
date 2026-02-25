import gnupg
import os
import zipfile
from Crypto.Cipher import AES
from Crypto.Random import get_random_bytes

class SecurityManager:
    def __init__(self, gpg_home="gpg_home"):
        self.gpg_home = gpg_home
        if not os.path.exists(gpg_home):
            os.makedirs(gpg_home)
        self.gpg = gnupg.GPG(gnupghome=gpg_home)
        
        # In a real system, we would import/generate a server key
        # For this demo, we'll generate a dummy key if none exists
        keys = self.gpg.list_keys()
        if not keys:
            input_data = self.gpg.gen_key_input(
                name_real="Project Phoenix Server",
                name_email="security@project-phoenix.ai",
                key_type="RSA",
                key_length=2048
            )
            self.gpg.gen_key(input_data)

    def sign_manifest(self, manifest_path):
        """
        Sign the JSON manifest with the server's PGP key.
        """
        with open(manifest_path, 'rb') as f:
            signed_data = self.gpg.sign_file(f, detach=True)
        
        signature_path = f"{manifest_path}.sig"
        with open(signature_path, 'w') as f:
            f.write(str(signed_data))
        
        return signature_path

    def encrypt_bundle(self, file_paths, bundle_output_path, password=None):
        """
        Create a ZIP bundle and (optionally) encrypt it with AES-256-GCM.
        Note: Section 65B bundle typically needs to be tamper-evident.
        """
        zip_path = f"{bundle_output_path}.zip"
        with zipfile.ZipFile(zip_path, 'w') as zipf:
            for file_path in file_paths:
                zipf.write(file_path, os.path.basename(file_path))
        
        if password:
            # Encrypt the ZIP file
            key = hashlib.sha256(password.encode()).digest()
            nonce = get_random_bytes(12)
            cipher = AES.new(key, AES.MODE_GCM, nonce=nonce)
            
            with open(zip_path, 'rb') as f:
                data = f.read()
            
            ciphertext, tag = cipher.encrypt_and_digest(data)
            
            enc_path = f"{zip_path}.enc"
            with open(enc_path, 'wb') as f:
                for x in (nonce, tag, ciphertext):
                    f.write(x)
            
            # Remove the unencrypted zip
            os.remove(zip_path)
            return enc_path
            
        return zip_path

    def delete_after_delay(self, file_path, delay=3600):
        """
        Mock for a background task to delete files after a certain delay.
        """
        # In production, use a task queue like Celery or a background thread
        pass
