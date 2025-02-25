from django.shortcuts import render
from django.views.generic import TemplateView #render a viewing platform that poses request to client
# Create your views here.
from servs.HypothesizeemojisScript import scripts

class Idxs(TemplateView):
    template_name = "Idxs.html"

    def post(self, request):
        content = request.POST["content"]
        emoji = scripts.hypothesis(content)

        cxts = {
            "content": content,
            "emoji": emoji
        }

        return render(request, self.template_name, context=cxts)