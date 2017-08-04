package controllers;

import play.mvc.*;
import play.data.*;
import static play.data.Form.*;

import views.html.*;

import models.*;

import crawler.*;


/**
 * Manage a database of computers
 */
public class Application extends Controller {
    
    /**
     * This result directly redirect to application home.
     */
    public static Result GO_HOME = redirect(
        routes.Application.index()
    );
    
    /**
     * Handle default path requests, redirect to computers list
     */
    public static Result index() {

    	Crawler crawler = new Crawler();
    	//crawler.StartCrawl("http://openaccess.thecvf.com/","CVPR2017.py");
    	//crawler.UpdateAbstract();
    	//crawler.UpdatePDFText();
    	crawler.OutputTitleWords("title_words.txt");
    	crawler.OutputTitles("title.txt");
    	crawler.OutputAbstracts("abstract.txt");
    	crawler.OutputCorpus("corpus.txt");
    	
        return ok( index.render( crawler.logMessage ) );
    }

    /**
     * Display the paginated list of computers.
     *
     * @param page Current page number (starts from 0)
     * @param sortBy Column to be sorted
     * @param order Sort order (either asc or desc)
     * @param filter Filter applied on computer names
     */
    public static Result list(int page, String sortBy, String order, String filter) {
        return GO_HOME;
    }
    
    /**
     * Display the 'edit form' of a existing Computer.
     *
     * @param id Id of the computer to edit
     */
    public static Result edit(Long id) {
        return GO_HOME;
    }
    
    /**
     * Handle the 'edit form' submission 
     *
     * @param id Id of the computer to edit
     */
    public static Result update(Long id) {
        return GO_HOME;
    }
    
    /**
     * Display the 'new computer form'.
     */
    public static Result create() {
        return GO_HOME;
    }
    
    /**
     * Handle the 'new computer form' submission 
     */
    public static Result save() {
        return GO_HOME;
    }
    
    /**
     * Handle computer deletion
     */
    public static Result delete(Long id) {

        return GO_HOME;
    }
    

}
            
