import boto3
import json
import urllib.parse
from datetime import datetime

rekognition = boto3.client('rekognition')
dynamodb = boto3.resource('dynamodb')
table = dynamodb.Table('ImageResults')


def lambda_handler(event, context):

    # ── Handle API Gateway GET request ──────────────────────────────
    if event.get('httpMethod') == 'GET':
        params = event.get('queryStringParameters') or {}
        image_key = params.get('imageKey')

        if not image_key:
            return {
                'statusCode': 400,
                'headers': {'Access-Control-Allow-Origin': '*'},
                'body': json.dumps({'error': 'imageKey parameter is required'})
            }

        result = table.get_item(Key={'imageKey': image_key})
        item = result.get('Item')

        if not item:
            return {
                'statusCode': 404,
                'headers': {'Access-Control-Allow-Origin': '*'},
                'body': json.dumps({'error': 'Image not found in database'})
            }

        return {
            'statusCode': 200,
            'headers': {'Access-Control-Allow-Origin': '*'},
            'body': json.dumps(item)
        }

    # ── Handle S3 trigger (image upload event) ──────────────────────
    try:
        record = event['Records'][0]
        bucket = record['s3']['bucket']['name']
        key = urllib.parse.unquote_plus(record['s3']['object']['key'])

        print(f"Processing image: {key} from bucket: {bucket}")

        # ── Step 1: Detect Labels ────────────────────────────────────
        label_response = rekognition.detect_labels(
            Image={'S3Object': {'Bucket': bucket, 'Name': key}},
            MaxLabels=10,
            MinConfidence=75
        )
        labels = [
            {
                'name': label['Name'],
                'confidence': round(label['Confidence'], 2)
            }
            for label in label_response['Labels']
        ]

        # ── Step 2: Detect Faces ─────────────────────────────────────
        face_response = rekognition.detect_faces(
            Image={'S3Object': {'Bucket': bucket, 'Name': key}},
            Attributes=['ALL']
        )
        faces = []
        for face in face_response['FaceDetails']:
            faces.append({
                'ageRange': face.get('AgeRange', {}),
                'gender': face.get('Gender', {}).get('Value', 'Unknown'),
                'smile': face.get('Smile', {}).get('Value', False),
                'emotions': [
                    e['Type'] for e in face.get('Emotions', [])
                    if e['Confidence'] > 50
                ]
            })

        # ── Step 3: Detect Text in image ─────────────────────────────
        text_response = rekognition.detect_text(
            Image={'S3Object': {'Bucket': bucket, 'Name': key}}
        )
        detected_text = [
            t['DetectedText']
            for t in text_response['TextDetections']
            if t['Type'] == 'LINE' and t['Confidence'] > 80
        ]

        # ── Step 4: Store results in DynamoDB ────────────────────────
        item = {
            'imageKey': key,
            'bucket': bucket,
            'uploadedAt': datetime.utcnow().isoformat(),
            'labels': labels,
            'labelNames': [l['name'] for l in labels],
            'faceCount': len(faces),
            'faces': faces,
            'detectedText': detected_text
        }

        table.put_item(Item=item)
        print(f"Saved results to DynamoDB: {len(labels)} labels, {len(faces)} faces")

        return {
            'statusCode': 200,
            'body': json.dumps({
                'message': 'Image processed successfully',
                'imageKey': key,
                'labels': [l['name'] for l in labels],
                'faceCount': len(faces),
                'detectedText': detected_text
            })
        }

    except Exception as e:
        print(f"ERROR: {str(e)}")
        return {
            'statusCode': 500,
            'body': json.dumps({'error': str(e)})
        }