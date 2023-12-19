import inference

model_name="klue/roberta-small"
output_sample_path='./data/3trance2xnolabel.csv'

if __name__ == "__main__":
  output = inference.pd.read_csv(output_sample_path)

  output['label'] = inference.inference(
    model_name=model_name,
    predict_path=output_sample_path,
    output_sample_path=output_sample_path)
  
  output.to_csv(f'./csv/{model_name}_pseudolabel.csv', index=False)