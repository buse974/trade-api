import pg from 'pg';
import dotenv from 'dotenv';

dotenv.config();

const pool = new pg.Pool({
  connectionString: process.env.DATABASE_URL || 'postgresql://trade:trade_secret@172.17.0.1:5432/trade',
});

export default pool;
