const { ApolloGateway } = require('@apollo/gateway');
const { ApolloServer } = require('@apollo/server');
const { startStandaloneServer } = require('@apollo/server/standalone');

const gateway = new ApolloGateway({
  serviceList: [
    { name: 'users', url: 'http://localhost:8001/graphql' },
    { name: 'products', url: 'http://localhost:8002/graphql' },
    { name: 'orders', url: 'http://localhost:8003/graphql' },
  ],
});

const server = new ApolloServer({
  gateway,
  subscriptions: false,
});

startStandaloneServer(server, {
  listen: { port: 4000 },
}).then(({ url }) => {
  console.log(`Gateway ready at ${url}`);
});
