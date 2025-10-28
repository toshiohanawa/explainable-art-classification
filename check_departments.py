import requests
import json

# 部門一覧を取得
url = 'https://collectionapi.metmuseum.org/public/collection/v1/departments'
response = requests.get(url)
data = response.json()

print('部門一覧:')
for dept in data['departments']:
    print(f'ID: {dept["departmentId"]}, 名前: {dept["displayName"]}')

print('\n絵画関連部門を確認:')
for dept in data['departments']:
    if 'paint' in dept['displayName'].lower() or 'european' in dept['displayName'].lower():
        print(f'ID: {dept["departmentId"]}, 名前: {dept["displayName"]}')

# 各部門の作品数を確認
print('\n各部門の作品数:')
for dept in data['departments']:
    dept_id = dept['departmentId']
    try:
        objects_url = f'https://collectionapi.metmuseum.org/public/collection/v1/objects?departmentIds={dept_id}'
        objects_response = requests.get(objects_url)
        objects_data = objects_response.json()
        count = len(objects_data.get('objectIDs', []))
        print(f'部門 {dept_id} ({dept["displayName"]}): {count}件')
    except Exception as e:
        print(f'部門 {dept_id} ({dept["displayName"]}): エラー - {e}')
