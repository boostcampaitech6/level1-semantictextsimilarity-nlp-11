import os

def send2slack(
        channel='',
        username='',
        emoji='',
        title='',
        hook='###############################여기에 slack incoming webhook에서 채널을 선택해서 웹후크를 생성하여 주소를 넣으면 됩니다###################################',
        msg='작업 완료알림'):

    PAYLOAD=f'\"channel\": \"{channel}\", \"username\": \"{username}\", \"text\": \"{title} \n\n {msg}\", \"icon_emoji\": \"{emoji}\"'
    PAYLOAD='payload={'+PAYLOAD+'}'

    os.system(f'/usr/bin/curl --data-urlencode \'{PAYLOAD}\' \'{hook}\'')

if __name__ == "__main__":
    send2slack()