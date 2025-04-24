// init-mongo.js
db = db.getSiblingDB('caas'); // creates or switches to 'mydatabase'
db.createCollection('caas'); // creates a collection
