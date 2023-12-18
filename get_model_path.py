import os

def get_best_checkpoint(checkpoint_dir):
  """
  폴더 안에 있는 .ckpt 파일 중 {val_pearson:.2f}이 가장 높은 파일을 반환합니다.

  Args:
      checkpoint_dir: 체크포인트 파일이 저장된 디렉토리

  Returns:
      가장 높은 {val_pearson:.2f}을 가진 체크포인트 파일
  """

  checkpoints = os.listdir(checkpoint_dir)
  best_checkpoint = None
  best_pearson = -float("inf")

  for checkpoint in checkpoints:
    if not checkpoint.endswith("pt"):
      continue

    # {val_pearson:.2f}을 추출합니다.
    try:
      pearson = float(checkpoint.replace("=","_").split("_")[-2])

    # Pearson 점수를 비교합니다.
      if pearson > best_pearson:
        best_pearson = pearson
        best_checkpoint = checkpoint
    except:
      pass  

  return best_checkpoint


def get_safe_filename(filename):
    """
    파일 이름에 포함되어 있으면 문제가 되는 문자를 안전한 문자로 치환합니다.

    Args:
        filename: 파일 이름

    Returns:
        안전한 파일 이름
    """

    safe_filename = filename

    for bad_char in ["/", "\\", "*", "'", '"', "**", ":", "?", "\"", "<", ">", "|"]:
        if bad_char in filename:
            safe_filename = safe_filename.replace(bad_char, "")

    return safe_filename