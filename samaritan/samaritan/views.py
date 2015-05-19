from django.views import generic
from django.http import HttpResponse

from samaritan import author_classifier

from pos_tagger import tag_text
import dataset


class Home(generic.TemplateView):
    template_name = 'index.html'


class Classify(generic.View):
    def get(self, request, *args, **kwargs):
        text = request.GET.get('q', '').strip()
        text.replace('\n', ' ')
        if len(text) > 0:
            tags = tag_text(text)
            features = dataset.__convern_to_count_dictionary(tags, n_gram=4)
            author = author_classifier.predict(features)
            return HttpResponse(author, status=200)
        return HttpResponse(status=400)
