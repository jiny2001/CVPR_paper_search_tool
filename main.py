import os
import jinja2
import webapp2
from model.paper2vec import Paper2Vec

PAGE_TITLE = "ICCV2017"

JINJA_ENVIRONMENT = jinja2.Environment(
	loader=jinja2.FileSystemLoader(os.path.dirname(__file__)),
	extensions=['jinja2.ext.autoescape'],
	autoescape=True)

class PaperSearchResults():

	def __init__(self, id, score, title, abstract_url, pdf_url):
		self.id = id
		self.score = int(score)
		self.title = title
		self.abstract_url = abstract_url
		self.pdf_url = pdf_url


class BaseHandler(webapp2.RequestHandler):
	def render(self, html, values={}):
		template = JINJA_ENVIRONMENT.get_template(html)
		self.response.write(template.render(values))

class MainPage(BaseHandler):

	def get(self):
		values = {
			'title': PAGE_TITLE,
			'searchResult': [],
			'message': None
		}
		self.render("html/main.html", values)

	def post(self):
		keywords_text = self.request.get('keywords')
		if keywords_text is None:
			self.redirect('/')

		keywords = keywords_text.split(" ")

		results = p2v.find_by_keywords(keywords)
		paperSearchResult = []

		for result in results:
			paperInfo = PaperSearchResults(result[0], result[1],
			                      p2v.paper[result[0]].title,
			                      p2v.paper[result[0]].abstract_url,
			                      p2v.paper[result[0]].pdf_url)
			paperSearchResult.append(paperInfo)

		if len(results) > 0:
			message = ("%d papers found. [ " + keywords_text +" ]") % len(results)
		else:
			message = "No papers found. [ " + keywords_text +" ]"
		values = {
			'title': PAGE_TITLE,
			'searchResult': paperSearchResult,
			'message': message
		}

		self.render("html/main.html", values)

class FindSimilarPaperPage(BaseHandler):

	def get(self):
		paper_id = int(self.request.get('paper_id'))

		if paper_id is None or paper_id < 0 or paper_id >= p2v.papers:
			self.redirect('/')

		target = PaperSearchResults(paper_id, 0,
			                      p2v.paper[paper_id].title,
			                      p2v.paper[paper_id].abstract_url,
			                      p2v.paper[paper_id].pdf_url)

		results = p2v.find_similar_papers(paper_id, 5)

		paperSearchResult = []
		for result in results:
			paperInfo = PaperSearchResults(result[0], result[1],
			                      p2v.paper[result[0]].title,
			                      p2v.paper[result[0]].abstract_url,
			                      p2v.paper[result[0]].pdf_url)
			paperSearchResult.append(paperInfo)

		values = {
			'title': PAGE_TITLE,
			'paper': target,
			'searchResult':paperSearchResult
		}
		self.render("html/paper.html", values)


p2v = Paper2Vec("data")
p2v.load_paper_vectors()

app = webapp2.WSGIApplication([
	('/', MainPage),
	('/find_similar_paper', FindSimilarPaperPage),
], debug=True)
