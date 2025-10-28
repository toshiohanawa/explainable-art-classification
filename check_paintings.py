import requests
import json

# European Paintings部門の作品を取得
url = 'https://collectionapi.metmuseum.org/public/collection/v1/objects?departmentIds=11'
response = requests.get(url)
data = response.json()

print(f'European Paintings部門の総作品数: {len(data["objectIDs"])}')

# 最初の10件の詳細を確認
print('\n最初の10件の詳細:')
for i, object_id in enumerate(data['objectIDs'][:10]):
    detail_url = f'https://collectionapi.metmuseum.org/public/collection/v1/objects/{object_id}'
    detail_response = requests.get(detail_url)
    detail_data = detail_response.json()
    
    object_name = detail_data.get('objectName', 'N/A')
    title = detail_data.get('title', 'N/A')
    is_public_domain = detail_data.get('isPublicDomain', False)
    
    try:
        print(f'{i+1}. ID: {object_id}, Name: {object_name}, Title: {title[:50]}..., Public: {is_public_domain}')
    except UnicodeEncodeError:
        print(f'{i+1}. ID: {object_id}, Name: {object_name}, Title: [文字化け], Public: {is_public_domain}')

# 絵画以外の作品を確認
print('\n絵画以外の作品を確認:')
non_painting_count = 0
for i, object_id in enumerate(data['objectIDs'][:50]):  # 最初の50件をチェック
    detail_url = f'https://collectionapi.metmuseum.org/public/collection/v1/objects/{object_id}'
    detail_response = requests.get(detail_url)
    detail_data = detail_response.json()
    
    object_name = detail_data.get('objectName', '').lower()
    if not any(painting_type in object_name for painting_type in ['painting', 'canvas', 'oil', 'watercolor', 'tempera']):
        non_painting_count += 1
        print(f'非絵画: ID {object_id}, Name: {detail_data.get("objectName", "N/A")}')

print(f'\n最初の50件中、絵画以外: {non_painting_count}件')
