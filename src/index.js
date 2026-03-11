import express from 'express';
import cors from 'cors';
import dotenv from 'dotenv';
import swaggerUi from 'swagger-ui-express';

dotenv.config();

const app = express();
const PORT = process.env.PORT || 3000;

// Swagger documentation
const swaggerDocument = {
  openapi: '3.0.0',
  info: {
    title: 'trade API',
    version: '1.0.0',
    description: 'Bot de trading SOL/USDC avec simulation et prédiction ML multi-crypto'
  },
  servers: [
    { url: 'http://localhost:3000', description: 'Local' },
    { url: 'https://api.trade.51.77.223.61.nip.io', description: 'Production' }
  ],
  paths: {
    '/health': {
      get: {
        summary: 'Health check',
        responses: { 200: { description: 'OK' } }
      }
    }
  }
};

app.use(cors());
app.use(express.json());

// Swagger UI route
app.use('/docs', swaggerUi.serve, swaggerUi.setup(swaggerDocument));

app.get('/health', (req, res) => {
  res.json({ status: 'ok', service: 'trade-api', type: 'api' });
});

app.listen(PORT, () => {
  console.log(`Server running on port ${PORT}`);
  console.log(`Swagger docs: http://localhost:${PORT}/docs`);
});
