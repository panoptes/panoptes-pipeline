import webapp2
import os
from google.appengine.ext.webapp import template

class MainPage(webapp2.RequestHandler):

    def get(self):
        pic = 'PIC_J18365633+3847012'
        image_name = '{}.png'.format(pic)
        caption = "PANOPTES Light Curve Dashboard"
        probability = "There is a XXX probability that an exoplanet is orbiting star {}.".format(pic)
        template_values = {'caption': caption, "probability": probability, 'image_name': image_name}
        path = os.path.join(os.path.dirname(__file__), 'index.html')
        self.response.out.write(template.render(path, template_values))


app = webapp2.WSGIApplication([
    ('/', MainPage),
], debug=True)
