# Write data in SQLite database
# '''
import sqlite3
from sqlite3 import Error
import pandas as pd

class CSQLLite:
    def __init__(self, db_file):
        self.db_file = db_file
        self.conn = self.create_connection()
        

    # This was also create file if it doesnt exist
    def create_connection(self):
        """ create a database connection to the SQLite database
            specified by db_file
        :param db_file: database file
        :return: Connection object or None
        """
        conn = None
        try:
            conn = sqlite3.connect(self.db_file)
            return conn
        except Error as e:
            print(e)

        return conn
    
    def execute_query(self, query, params=()):
        """ Execute a single query
        :param query: a SQL query
        :param params: parameters to the SQL query
        :return: True if successful, False otherwise
        """
        try:
            c = self.conn.cursor()
            c.execute(query, params)
            result = c.lastrowid
            self.conn.commit()
            return result
        except Error as e:
            print(f"Error executing query: {e}")
            return -1
        
    def execute_read_query(self, query, params=()):
        """ Execute a read query and return the results
        :param query: a SQL query
        :param params: parameters to the SQL query
        :return: list of tuples containing the results
        """
        try:
            c = self.conn.cursor()
            c.execute(query, params)
            result = c.fetchall()
            return result
        except Error as e:
            print(f"Error executing read query: {e}")
            return []