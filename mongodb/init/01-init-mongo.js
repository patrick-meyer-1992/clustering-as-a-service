// init-mongo.js
db = db.getSiblingDB('caas'); // creates or switches to 'caas'
db.createCollection('data'); // creates a collection
db.createCollection('results'); // creates a collection
