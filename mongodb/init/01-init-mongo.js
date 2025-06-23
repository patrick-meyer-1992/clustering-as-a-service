// init-mongo.js
db = db.getSiblingDB('caas'); // creates or switches to 'caas'
db.createUser({ // SK - creates a user automatically for hard reset
  user: "admin",
  pwd: "password123",
  roles: [{ role: "readWrite", db: "caas" }]
});
db.createCollection('data'); // creates a collection
db.createCollection('results'); // creates a collection
