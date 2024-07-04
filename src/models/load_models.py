import torch
import torch.nn as nn
from transformers import (
    AutoImageProcessor,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    BeitFeatureExtractor,
    BertModel,
    BertTokenizer,
    Data2VecVisionForImageClassification,
    Data2VecVisionModel,
    ResNetForImageClassification,
    RobertaModel,
    RobertaTokenizer,
    ViTForImageClassification,
    ViTImageProcessor,
    ViTModel,
)


def load_model(name, data_type, device):
    model = Classifier(name, data_type, device)
    return model


def load_transformation(name: str):
    """
    Loads transformation specified by name of the model for which the data will be used.
    :param name: Name of the model
    :return: Transformation function of data.
    """
    if name == "resnet50":
        transform = AutoImageProcessor.from_pretrained("microsoft/resnet-50")
    elif name == "vit":
        transform = ViTImageProcessor.from_pretrained("google/vit-base-patch16-224-in21k")
    elif name == "vit_finetuned":
        transform = ViTImageProcessor.from_pretrained("google/vit-base-patch16-224")
    elif name == "data2vec":
        transform = AutoImageProcessor.from_pretrained("facebook/data2vec-vision-base")
    elif name == "data2vec_finetuned":
        transform = BeitFeatureExtractor.from_pretrained("facebook/data2vec-vision-base-ft1k")
    elif name == "roberta":
        transform = RobertaTokenizer.from_pretrained("roberta-base")
    elif name == "roberta_go":
        transform = AutoTokenizer.from_pretrained("SamLowe/roberta-base-go_emotions")
    elif name == "bert":
        transform = BertTokenizer.from_pretrained("bert-base-uncased")
    elif name == "bert_finetuned":
        transform = BertTokenizer.from_pretrained("bhadresh-savani/bert-base-uncased-emotion")
    else:
        raise NotImplementedError(f"Transformation for model {name} not implemented.")
    return transform


class Classifier(nn.Module):
    def __init__(self, name: str, data_type: str, device):
        super(Classifier, self).__init__()
        self.name = name
        self.data_type = data_type
        self.device = device
        if name == "resnet50":
            self.model = ResNetForImageClassification.from_pretrained("microsoft/resnet-50")
        elif name == "vit":
            self.model = ViTModel.from_pretrained("google/vit-base-patch16-224-in21k", add_pooling_layer=False)
        elif name == "vit_finetuned":
            self.model = ViTForImageClassification.from_pretrained("google/vit-base-patch16-224")
        elif self.name == "data2vec":
            self.model = Data2VecVisionModel.from_pretrained("facebook/data2vec-vision-base", add_pooling_layer=True)
        elif self.name == "data2vec_finetuned":
            self.model = Data2VecVisionForImageClassification.from_pretrained("facebook/data2vec-vision-base-ft1k")
        elif name == "roberta_go":
            self.model = AutoModelForSequenceClassification.from_pretrained("SamLowe/roberta-base-go_emotions")
        elif name == "roberta":
            self.model = RobertaModel.from_pretrained("roberta-base")
        elif name == "bert":
            self.model = BertModel.from_pretrained("bert-base-uncased")
        elif name == "bert_finetuned":
            self.model = BertModel.from_pretrained("bhadresh-savani/bert-base-uncased-emotion")
        else:
            raise NotImplementedError("Model {} not implemented".format(name))
        self.model = self.model.to(self.device)
        self.transform = load_transformation(self.name)

    def input_to_representation(self, x, layer=12):
        if self.data_type == "images":
            x = self.transform(x, return_tensors="pt")
        elif self.data_type == "text":
            x = self.transform(x, padding=True, return_tensors="pt", truncation=True, max_length=512)
        else:
            raise NotImplementedError
        x = x.to(self.device)
        if self.name == "resnet50":
            outputs = self.model.resnet(**x, output_hidden_states=False, return_dict=True)
            pooled_output = outputs.pooler_output
        elif self.name in [
            "vit",
            "vit_finetuned",
            "data2vec",
            "data2vec_finetuned",
            "roberta_go",
            "roberta",
            "bert",
            "bert_finetuned",
        ]:
            outputs = self.model(**x, output_hidden_states=True, return_dict=True)
            pooled_output = torch.mean(outputs.hidden_states[layer], dim=1)
        else:
            raise NotImplementedError("Model {} not implemented".format(self.name))
        return pooled_output

    def representation_to_output(self, h):
        logits = self.classifier(h)
        return logits
