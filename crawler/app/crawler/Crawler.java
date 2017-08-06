package crawler;

import controllers.Application;
import models.*;

import play.Logger;
import play.Logger.ALogger;

import java.io.BufferedWriter;
import java.io.File;
import java.io.FileWriter;
import java.net.URL;
import java.util.*;
import com.gargoylesoftware.htmlunit.*;
import com.gargoylesoftware.htmlunit.html.HtmlAnchor;
import com.gargoylesoftware.htmlunit.html.HtmlElement;
import com.gargoylesoftware.htmlunit.html.HtmlPage;


import org.apache.pdfbox.pdmodel.*;
import org.apache.pdfbox.text.*;

public class Crawler {

	private static final ALogger logger = Logger.of(Application.class);
	
	public String BaseURL;
	public String Target;
	public String logMessage;
	List<?> Elements;
	
	public boolean StartCrawl(String baseUrl, String target)
	{
		BaseURL = baseUrl;
		Target = target;
		logMessage = "StartCrawling...<br/>";
		
		WebClient webClient = new WebClient(BrowserVersion.CHROME);
		
		try
		{
			HtmlPage Page = webClient.getPage( BaseURL + Target );
		
			logMessage += "All Titles: ";

			Elements = Page.getByXPath( "/html/body/div[@id='content']/dl/dt[@class='ptitle']/a" );
			if( Elements == null )
				logMessage += "0" + "<br/>";
			else
			{
				logMessage += Integer.toString( Elements.size() ) + "<br/>";
				
				for( int i = 0; i < Elements.size(); i++ )
				{
					HtmlAnchor a = (HtmlAnchor) Elements.get(i);
					logMessage += a.getTextContent() + " ";
					logMessage += BaseURL + a.getHrefAttribute() + "<br/>";

					Paper paper;
					Long id = new Long(i+1);
					paper = Paper.find.byId(id);
					if( paper == null )
					{
						paper = new Paper();
						paper.id = id;
						paper.save();
					}
					paper.title = a.getTextContent();
					paper.abstract_url = BaseURL + a.getHrefAttribute();
					paper.update();
				}
			}
			
			
			logMessage += "All PDF links: ";
			
			Elements = Page.getByXPath( "//A[contains(text(), 'pdf')]" );
			if( Elements == null )
				logMessage += "0" + "<br/>";
			else
			{
				logMessage += Integer.toString( Elements.size() ) + "<br/>";
				
				for( int i = 0; i < Elements.size(); i++ )
				{
					HtmlAnchor a = (HtmlAnchor) Elements.get(i);
					logMessage += BaseURL + a.getHrefAttribute() + "<br/>";

					Paper paper;
					Long id = new Long(i+1);
					paper = Paper.find.byId(id);
					if( paper == null )
					{
						paper = new Paper();
						paper.id = id;
						paper.save();
					}
					paper.pdf_url = BaseURL + a.getHrefAttribute();
					paper.update();

				}
			}
		}
		catch(Exception e)
		{
			logMessage += "Error:" + e.getMessage();
			e.printStackTrace();
		}
		
		webClient.close();

		return true;
	}
	
	public boolean UpdateAbstract()
	{
		WebClient webClient = new WebClient(BrowserVersion.CHROME);
		logMessage = "UpdateAbstract:<br/>";
		
		try
		{
			List<Paper> papers = Paper.list();
			
			for (int i = 0; i < papers.size(); i++) 
			{
				if(papers.get(i).abstract_text!= null && papers.get(i).abstract_text.length()>=1)
    			    continue;

				HtmlPage Page = webClient.getPage( papers.get(i).abstract_url );

				HtmlElement element = Page.getFirstByXPath( "/html/body/div[@id='content']/dl/dd/div[@id='abstract']" );
				if( element != null)
				{
					papers.get(i).abstract_text = element.getTextContent();
					papers.get(i).update();
					logger.info("updated abstract_text:" + i);
				}	
			}
		}
		catch(Exception e)
		{
			logMessage += "Error:" + e.getMessage();
			e.printStackTrace();
		}
		
		webClient.close();
		
		return true;
	}

	public boolean UpdatePDFText()
	{
		WebClient webClient = new WebClient(BrowserVersion.CHROME);
		logMessage = "UpdatePDFText:<br/>";
		logger.info("updating pdf text...");
		try
		{
			List<Paper> papers = Paper.list();
			
			for (int i = 0; i < papers.size(); i++) 
			{
				Paper paper = papers.get(i);

				if( paper.pdf_text==null ||paper.pdf_text.length() <= 1)
				{
					try
					{
						URL u = new URL(paper.pdf_url);
					    PDDocument pddDocument = PDDocument.load(u.openStream() );
					    try
					    {
						    PDFTextStripper textStripper = new PDFTextStripper();
						    String doc = textStripper.getText(pddDocument);
							paper.pdf_text = doc;
							paper.update();
							logger.info("updated abstract_text:" + i);
					    }
						finally
						{
						    pddDocument.close();
						}
					}
					catch(Exception e)
					{
						logMessage += "Error:" + e.getMessage();
						e.printStackTrace();
					}
				}
		    }
		}
		catch(Exception e)
		{
			logMessage += "Error:" + e.getMessage();
			e.printStackTrace();
		}
		
		webClient.close();
		
		return true;
	}
	
	public String NormalizeText(String text, boolean with_eos)
	{
	    //replace "3D" before any number is removed
		text = text.replaceAll("[^a-zA-Z]3D[^a-zA-Z]"," Three Dimensional ");
		
	    //fix hyphenated words
 		text = text.replaceAll("-\\r\\n","");
 		text = text.replaceAll("-\\r","");
 		text = text.replaceAll("-\\n","");
 		text = text.replaceAll("\\r","");
 		text = text.replaceAll("\\n","");
 		
 		//removing numbers or other control characters
		text = text.replaceAll("[^a-zA-Z. ]+"," ");

		text = text.toLowerCase();
		
		// removing one length word
		text = text.replaceAll("\\b\\w{1}\\b", " ");

		// removing urls
		text = text.replaceAll("\\b(https?|ftp)[a-z]*"," ");


		// removing others
		text = text.replaceAll("\\bthe\\b", " ");
		text = text.replaceAll("\\ban\\b", " ");
		text = text.replaceAll("\\bin\\b", " ");
		text = text.replaceAll("\\bon\\b", " ");
		text = text.replaceAll("\\band\\b", " ");
		text = text.replaceAll("\\bof\\b", " ");
		text = text.replaceAll("\\bto\\b", " ");
		text = text.replaceAll("\\bis\\b", " ");
		text = text.replaceAll("\\bfor\\b", " ");
		text = text.replaceAll("\\bwe\\b", " ");
		text = text.replaceAll("\\bwith\\b", " ");
		text = text.replaceAll("\\bas\\b", " ");
		text = text.replaceAll("\\bthat\\b", " ");
		text = text.replaceAll("\\bare\\b", " ");
		text = text.replaceAll("\\bby\\b", " ");
		text = text.replaceAll("\\bour\\b", " ");
		text = text.replaceAll("\\bthis\\b", " ");
		text = text.replaceAll("\\bfrom\\b", " ");
		text = text.replaceAll("\\bbe\\b", " ");
		text = text.replaceAll("\\bcan\\b", " ");
		text = text.replaceAll("\\bat\\b", " ");
		text = text.replaceAll("\\bus\\b", " ");
		text = text.replaceAll("\\bit\\b", " ");
		text = text.replaceAll("\\bhas\\b", " ");
		text = text.replaceAll("\\bhave\\b", " ");
		text = text.replaceAll("\\bbeen\\b", " ");
		text = text.replaceAll("\\bdo\\b", " ");
		text = text.replaceAll("\\bdoes\\b", " ");
		text = text.replaceAll("\\bthese\\b", " ");
		text = text.replaceAll("\\bthose\\b", " ");
		text = text.replaceAll("\\bet\\b", " ");
		text = text.replaceAll("\\bal\\b", " ");

		// change popular plural noun to singular noun
		text = text.replaceAll("\\bpoints\\b", " point ");
		text = text.replaceAll("\\bimages\\b", " image ");
		text = text.replaceAll("\\bobjects\\b", " object ");
		text = text.replaceAll("\\blayers\\b", " layer ");
		text = text.replaceAll("\\bsamples\\b", " sample ");
		text = text.replaceAll("\\bfeatures\\b", " feature ");
		text = text.replaceAll("\\bsets\\b", " set ");
		text = text.replaceAll("\\bnetworks\\b", " network ");


    	text = text.replaceAll(" {2,}", " ");
	    text = text.replaceAll("\\.{2,}", ".");
		text = text.replaceAll("(\\. )+", ".");
		text = text.replaceAll("( \\.)+", ".");

		// removing people's name
		text = text.replaceAll("\\.([a-zA-Z]+\\.)+", ".");
		text = text.replaceAll("\\.([a-zA-Z]+ [a-zA-Z]+\\.)+", ".");

        if( with_eos )
    		text = text.replaceAll("\\.+", " <eos> ");
    	else
    		text = text.replaceAll("\\.+", " ");

    	text = text.replaceAll(" {2,}", " ");
		text = text.trim() + " ";

		return text;
	}
	
	public boolean OutputTitleWords(String filename)
	{
		logMessage = "OutputTitleWords:<br/>";
		logger.info("OutputTitleWords()...");

		try
		{
			File corpus=new File(filename);
			BufferedWriter writer = new BufferedWriter(new FileWriter(corpus));

			List<Paper> papers = Paper.list();
			
			for (int i = 0; i < papers.size(); i++) 
			{
				Paper paper = papers.get(i);

				if( paper.pdf_text!=null && paper.pdf_text.length() > 0 )
				{
				    writer.write(NormalizeText(paper.title, false) +" <eop> ");
				}
		    }

			writer.close();
		}
		catch(Exception e)
		{
			logMessage += "Error:" + e.getMessage();
			e.printStackTrace();
		}
	    
		return true;
	}
	
	public boolean OutputTitles(String filename)
	{
		logMessage = "OutputTitle:<br/>";
		logger.info("OutputTitle()...");

		try
		{
			File corpus=new File(filename);
			BufferedWriter writer = new BufferedWriter(new FileWriter(corpus));

			List<Paper> papers = Paper.list();
			
			for (int i = 0; i < papers.size(); i++) 
			{
				Paper paper = papers.get(i);

				if( paper.pdf_text!=null && paper.pdf_text.length() > 0 )
				{
				    writer.write (paper.title.trim()+" <eop> ");
				}
		    }

			writer.close();
		}
		catch(Exception e)
		{
			logMessage += "Error:" + e.getMessage();
			e.printStackTrace();
		}
	    
		return true;
	}

	public boolean OutputAbstracts(String filename)
	{
		logMessage = "OutputAbstracts:<br/>";
		logger.info("OutputAbstracts()...");

		try
		{
			File corpus=new File(filename);
			BufferedWriter writer = new BufferedWriter(new FileWriter(corpus));

			List<Paper> papers = Paper.list();
			
			for (int i = 0; i < papers.size(); i++) 
			{
				Paper paper = papers.get(i);

				if( paper.pdf_text!=null && paper.pdf_text.length() > 0 )
				{
				    writer.write (NormalizeText(paper.abstract_text, true)+"<eop> ");
				}
		    }

			writer.close();
		}
		catch(Exception e)
		{
			logMessage += "Error:" + e.getMessage();
			e.printStackTrace();
		}
	    
		return true;
	}
	
	public boolean OutputCorpus(String filename)
	{
		logMessage = "OutputCorpus:<br/>";
		logger.info("OutputCorpus()...");

		try
		{
			File corpus=new File(filename);
			BufferedWriter writer = new BufferedWriter(new FileWriter(corpus));

			List<Paper> papers = Paper.list();
			
 			for (int i = 0; i < papers.size(); i++) 
			{
				Paper paper = papers.get(i);
			    logger.info("corpus added for "+i);

				if( paper.pdf_text != null && paper.pdf_text.length() > 0 )
				    writer.write (NormalizeText(paper.pdf_text, true));
		    }

			writer.close();
		}
		catch(Exception e)
		{
			logMessage += "Error:" + e.getMessage();
			e.printStackTrace();
		}
	    
		return true;
	}
}
