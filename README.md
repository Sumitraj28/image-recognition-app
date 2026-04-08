# AWS Image Recognition App

A serverless image analysis app built on AWS.

## Architecture
User → S3 → Lambda → Rekognition → DynamoDB → API Gateway

## Tech Stack
- AWS S3 — image storage
- AWS Lambda — serverless backend (Python)
- AWS Rekognition — AI label & face detection
- AWS DynamoDB — results storage
- AWS API Gateway — REST API

## Files
- `frontend/index.html` — upload UI
- `frontend/script.js` — frontend logic
- `lambda/handler.py` — Lambda function

## How to Run
1. Deploy Lambda from `lambda/handler.py`
2. Set your S3 bucket name and API URL in `script.js`
3. Open `index.html` in browser

## Author
Raj Sumit