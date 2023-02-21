
// Load the TensorFlow.js library
import * as tf from '@tensorflow/tfjs';
import * as toxicity from '@tensorflow-models/toxicity';

// Load the model and specify the labels to check for
const model = await toxicity.load(0.9, ['toxicity']);

// Define a function to perform sentiment analysis
async function predictSentiment(text) {
  const predictions = await model.classify(text);
  const toxicityPrediction = predictions[0].results[0].match;
  return { text, toxicityPrediction };
}

// Call the function with a piece of text
const result = await predictSentiment('I hate this product. It is terrible.');
console.log(result.toxicityPrediction); // outputs true
