// API endpoint configuration
// Local development: http://127.0.0.1:5001
// Production: update PROD_API_URL to your Hugging Face Space URL
//   e.g. https://YOUR-HF-USERNAME-l2e-api.hf.space

const PROD_API_URL = 'https://YOUR-HF-USERNAME-l2e-api.hf.space';

const API = (
  window.location.hostname === '127.0.0.1' ||
  window.location.hostname === 'localhost' ||
  window.location.protocol === 'file:'
) ? 'http://127.0.0.1:5001' : PROD_API_URL;
