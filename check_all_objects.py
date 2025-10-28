import requests
import json

# 全部門から作品を取得
url = 'https://collectionapi.metmuseum.org/public/collection/v1/objects'
response = requests.get(url)
data = response.json()

print(f'全部門の総作品数: {len(data["objectIDs"])}')

# 最初の10件の詳細を確認
print('\n最初の10件の詳細:')
for i, object_id in enumerate(data['objectIDs'][:10]):
    detail_url = f'https://collectionapi.metmuseum.org/public/collection/v1/objects/{object_id}'
    detail_response = requests.get(detail_url)
    detail_data = detail_response.json()
    
    object_name = detail_data.get('objectName', 'N/A')
    title = detail_data.get('title', 'N/A')
    department = detail_data.get('department', 'N/A')
    is_public_domain = detail_data.get('isPublicDomain', False)
    
    try:
        print(f'{i+1}. ID: {object_id}, Name: {object_name}, Dept: {department}, Title: {title[:30]}..., Public: {is_public_domain}')
    except UnicodeEncodeError:
        print(f'{i+1}. ID: {object_id}, Name: {object_name}, Dept: {department}, Title: [文字化け], Public: {is_public_domain}')

# 絵画作品の数を確認
print('\n絵画作品の数を確認中...')
painting_count = 0
sample_paintings = []

for i, object_id in enumerate(data['objectIDs'][:1000]):  # 最初の1000件をチェック
    if i % 100 == 0:
        print(f'進捗: {i}/1000')
    
    detail_url = f'https://collectionapi.metmuseum.org/public/collection/v1/objects/{object_id}'
    try:
        detail_response = requests.get(detail_url)
        detail_response.raise_for_status()
        detail_data = detail_response.json()
    except Exception as e:
        print(f'エラー (ID: {object_id}): {e}')
        continue
    
    object_name = detail_data.get('objectName', '').lower()
    if any(painting_type in object_name for painting_type in ['painting', 'canvas', 'oil', 'watercolor', 'tempera']):
        painting_count += 1
        if len(sample_paintings) < 5:
            sample_paintings.append({
                'id': object_id,
                'name': detail_data.get('objectName', 'N/A'),
                'department': detail_data.get('department', 'N/A'),
                'title': detail_data.get('title', 'N/A')
            })

print(f'\n最初の1000件中、絵画作品: {painting_count}件')
print('\n絵画作品の例:')
for painting in sample_paintings:
    try:
        print(f'ID: {painting["id"]}, Name: {painting["name"]}, Dept: {painting["department"]}, Title: {painting["title"][:50]}...')
    except UnicodeEncodeError:
        print(f'ID: {painting["id"]}, Name: {painting["name"]}, Dept: {painting["department"]}, Title: [文字化け]')
