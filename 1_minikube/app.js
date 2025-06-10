const express = require('express');
const app = express();
const PORT = 3000;

app.get('/', (req, res) => {
  res.send('Hello Kubernetes!');
});

app.get('/metrics', (req, res) => {
  res.set('Content-Type', 'text/plain');
  res.send('# HELP node_app_info Simple Node.js app\n# TYPE node_app_info gauge\nnode_app_info 1');
});

app.listen(PORT, () => {
  console.log(`Server running on http://localhost:${PORT}`);
});