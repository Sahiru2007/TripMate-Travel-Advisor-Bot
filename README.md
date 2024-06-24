Certainly! Below is a detailed `README.md` for the AI-TravelBot TripMate project that includes a comprehensive description, features, installation steps, API configuration, usage, and additional information.


# AI-TravelBot TripMate

## Table of Contents
- [Project Description](#project-description)
- [Features](#features)
- [Technologies Used](#technologies-used)
- [Installation](#installation)
- [Configuration](#configuration)
  - [Twilio Setup](#twilio-setup)
  - [Stripe Setup](#stripe-setup)
  - [Google Maps API Setup](#google-maps-api-setup)
  - [OpenWeather API Setup](#openweather-api-setup)
  - [Amadeus API Setup](#amadeus-api-setup)
  - [OpenAI Setup](#openai-setup)
- [Usage](#usage)
- [System Architecture](#system-architecture)
  - [Architecture Diagram](#architecture-diagram)
- [Contributing](#contributing)
- [License](#license)

## Project Description
AI-TravelBot TripMate is an advanced travel assistant chatbot designed to provide users with comprehensive travel information, booking capabilities, and real-time updates. The chatbot integrates multiple APIs to offer functionalities such as flight searches, hotel recommendations, weather updates, and more.

## Features
- **Voice Recognition and Synthesis**: 
  - Converts speech to text using OpenAI's Whisper model.
  - Converts text responses to speech using Google Text-to-Speech (gTTS) API.
- **Real-Time Travel Data**:
  - Flight schedules, availability, pricing, hotel information, and car rentals via Amadeus API.
  - Local attractions and dining options via Google Places API.
  - Real-time weather updates via OpenWeather API.
- **Natural Language Processing (NLP)**:
  - Processes user messages and extracts key information using spaCy.
- **Emotion Classification**:
  - Predicts basic emotions based on user text input to tailor responses accordingly.
- **Dynamic Answer Generation**:
  - Fetches real-time data to generate dynamic responses.
- **Self-Training and Knowledge Base**:
  - Learns from user interactions to improve over time.

## Technologies Used
- **Frontend**: HTML, CSS, JavaScript
- **Backend**: Flask, Python
- **Database**: JSON-based knowledge base
- **APIs**: OpenAI, Amadeus, Google Places, OpenWeather
- **NLP**: spaCy
- **Voice Recognition and Synthesis**: Google Text-to-Speech, OpenAI Whisper

## Installation

### Prerequisites
- Python 3.7+
- Node.js and npm (for frontend development)

### Steps
1. **Clone the repository**
   ```bash
   git clone https://github.com/Sahiru2007/ai-travelbot-tripmate.git
   cd ai-travelbot-tripmate
   ```

2. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Setup Environment Variables**
   Create a `.env` file in the root directory and add the following variables:
   ```env
   FLASK_ENV=development
   TWILIO_ACCOUNT_SID=your_twilio_account_sid
   TWILIO_AUTH_TOKEN=your_twilio_auth_token
   TWILIO_PHONE_NUMBER=your_twilio_phone_number
   STRIPE_SECRET_KEY=your_stripe_secret_key
   GOOGLE_MAPS_API_KEY=your_google_maps_api_key
   OPENWEATHER_API_KEY=your_openweather_api_key
   AMADEUS_API_KEY=your_amadeus_api_key
   AMADEUS_API_SECRET=your_amadeus_api_secret
   OPENAI_API_KEY=your_openai_api_key
   ```

## Configuration

### Twilio Setup
1. **Sign Up**: Go to [Twilio](https://www.twilio.com/) and create an account.
2. **Get Credentials**: After signing up, get your Account SID and Auth Token from the Twilio Console.
3. **Phone Number**: Purchase a Twilio phone number from which SMS will be sent.
4. **Add to .env**: Add `TWILIO_ACCOUNT_SID`, `TWILIO_AUTH_TOKEN`, and `TWILIO_PHONE_NUMBER` to your `.env` file.

### Stripe Setup
1. **Sign Up**: Go to [Stripe](https://stripe.com/) and create an account.
2. **Get Secret Key**: From the Stripe Dashboard, get your Secret Key.
3. **Add to .env**: Add `STRIPE_SECRET_KEY` to your `.env` file.

### Google Maps API Setup
1. **Sign Up**: Go to [Google Cloud Platform](https://cloud.google.com/) and create a project.
2. **Enable APIs**: Enable the Maps JavaScript API and Places API for your project.
3. **Get API Key**: From the Google Cloud Console, get your API Key.
4. **Add to .env**: Add `GOOGLE_MAPS_API_KEY` to your `.env` file.

### OpenWeather API Setup
1. **Sign Up**: Go to [OpenWeather](https://openweathermap.org/) and create an account.
2. **Get API Key**: From the OpenWeather Dashboard, get your API Key.
3. **Add to .env**: Add `OPENWEATHER_API_KEY` to your `.env` file.

### Amadeus API Setup
1. **Sign Up**: Go to [Amadeus](https://developers.amadeus.com/) and create an account.
2. **Get Credentials**: From the Amadeus Dashboard, get your API Key and Secret.
3. **Add to .env**: Add `AMADEUS_API_KEY` and `AMADEUS_API_SECRET` to your `.env` file.

### OpenAI Setup
1. **Sign Up**: Go to [OpenAI](https://openai.com/) and create an account.
2. **Get API Key**: From the OpenAI Dashboard, get your API Key.
3. **Add to .env**: Add `OPENAI_API_KEY` to your `.env` file.

## Usage
1. **Start the Flask Server**
   ```bash
   flask run
   ```

2. **Open the Application**
   Go to `http://localhost:5000` in your browser.

3. **Interact with the Chatbot**
   Use the chatbot interface to ask travel-related questions, make bookings, and get real-time updates.

## System Architecture

### Architecture Diagram
<img width="740" alt="Screenshot 2024-06-24 at 23 05 40" src="https://github.com/Sahiru2007/TripMate-Travel-Advisor-Bot/assets/75121314/60fcd199-a2bd-4098-bd0f-4fe2a69f2848">

### Key Components
- **Frontend**: Provides the user interface for interacting with the chatbot.
- **Backend**: Handles API requests, processes user inputs, and manages the knowledge base.
- **APIs**: Integrates various third-party services for data and functionality.

## Contributing
Contributions are welcome! Please fork the repository and create a pull request with your changes. Ensure that your code follows the project's coding standards and includes appropriate tests.

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more details.
```

