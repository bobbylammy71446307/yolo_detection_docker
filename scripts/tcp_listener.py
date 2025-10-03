import socket
import json
import threading
import time

class TCP_Listener():

    def __init__(self, HOST='192.168.9.101', PORT=8080):
        self.HOST = HOST
        self.PORT = PORT
        self.PathName = ""
        self.running = False
        self.thread = None
        self.socket = None
        self.socket_lock = threading.Lock()

    def start(self):
        """Start the listener in a separate thread"""
        self.running = True
        self.thread = threading.Thread(target=self.client_listener)
        self.thread.start()
        print(f'TCP client connecting to {self.HOST}:{self.PORT}')
    
    def stop(self):
        """Stop the listener"""
        self.running = False

        # Close the socket to interrupt any blocking operations
        with self.socket_lock:
            if self.socket:
                try:
                    self.socket.close()
                    print("Disconnected from server")
                except:
                    pass
                self.socket = None

        if self.thread:
            self.thread.join(timeout=2)
        print("TCP listener stopped")

    def client_listener(self):
        """Connect to remote TCP server as a client"""
        while self.running:
            try:
                s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

                with self.socket_lock:
                    self.socket = s

                s.connect((self.HOST, self.PORT))
                print(f"Connected to server at {self.HOST}:{self.PORT}")
                s.settimeout(1.0)

                while self.running:
                    try:
                        data = s.recv(1024)
                        if not data:
                            print("Server closed the connection")
                            break

                        received_msg = data.decode("utf-8", errors="ignore")
                        print(f'Received from server: {received_msg}')

                        # Process the message
                        self.extract_pathname(received_msg)

                    except socket.timeout:
                        continue
                    except Exception as e:
                        if self.running:
                            print(f"Error receiving data: {e}")
                        break

            except ConnectionRefusedError:
                if self.running:
                    print(f"Connection refused by {self.HOST}:{self.PORT}. Retrying in 5 seconds...")
                    time.sleep(5)
            except Exception as e:
                if self.running:
                    print(f"Connection error: {e}. Retrying in 5 seconds...")
                    time.sleep(5)
            finally:
                with self.socket_lock:
                    if self.socket:
                        try:
                            self.socket.close()
                        except:
                            pass
                        self.socket = None

    def extract_pathname(self, message):
        """Extract pathname from JSON message, return True if successful"""
        try:
            data_dict = json.loads(message)
            
            if "InPathName" in data_dict:
                self.PathName = data_dict["InPathName"]
                print(f"PathName set to: {self.PathName}")
                return True
            else:
                print("Warning: 'InPathName' key not found in JSON message")
                return False
                
        except json.JSONDecodeError:
            print("Error: Received message is not valid JSON")
            return False
        except Exception as e:
            print(f"Error extracting pathname: {e}")
            return False


if __name__ == "__main__":
    t = TCP_Listener()
    t.start()
    try:
        while True:
            print(f"Current PathName: {t.PathName}")
            time.sleep(5)
    except KeyboardInterrupt:
        print("\nShutting down...")
        t.stop()