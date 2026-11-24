// Logging for the UI's scripts: every record at or above the file level
// goes to a log file, and the terminal shows those at or above the console
// level. Levels: trace, debug, info, warn, error; both default to info.
//
// Options every script takes: --log-level, --console-level, --log-file.
import fs from 'node:fs';
import path from 'node:path';

export const LEVELS = [ 'trace', 'debug', 'info', 'warn', 'error' ];

function rank( level ) {
  const index = LEVELS.indexOf( level );
  if ( index < 0 ) {
    throw new Error( `unknown level ${level}; use one of ${LEVELS.join( ', ' )}` );
  }
  return index;
}

// Splits the logging options from the rest of argv.
export function parseLogArgs( argv ) {
  const options = { logLevel: 'info', consoleLevel: 'info', logFile: null, rest: [] };
  for ( let i = 0; i < argv.length; i += 1 ) {
    const [ flag, inline ] = argv[ i ].split( /=(.*)/s, 2 );
    const take = () => inline ?? argv[ ( i += 1 ) ];
    if ( flag === '--log-level' ) {
      options.logLevel = take();
    } else if ( flag === '--console-level' ) {
      options.consoleLevel = take();
    } else if ( flag === '--log-file' ) {
      options.logFile = take();
    } else {
      options.rest.push( argv[ i ] );
    }
  }
  rank( options.logLevel );
  rank( options.consoleLevel );
  return options;
}

function stamp() {
  const now = new Date();
  const pad = ( n, width = 2 ) => String( n ).padStart( width, '0' );
  return `${now.getFullYear()}-${pad( now.getMonth() + 1 )}-${pad( now.getDate() )} `
    + `${pad( now.getHours() )}:${pad( now.getMinutes() )}:${pad( now.getSeconds() )},${pad( now.getMilliseconds(), 3 )}`;
}

export function createLogger( name, { logFile, logLevel = 'info', consoleLevel = 'info' } ) {
  fs.mkdirSync( path.dirname( logFile ), { recursive: true } );
  const fd = fs.openSync( logFile, 'a' );
  const fileRank = rank( logLevel );
  const consoleRank = rank( consoleLevel );
  const write = ( level, message ) => {
    const line = `${stamp()} [${level.toUpperCase().padEnd( 5 )}] ${name}: ${message}\n`;
    if ( rank( level ) >= fileRank ) {
      fs.writeSync( fd, line );
    }
    if ( rank( level ) >= consoleRank ) {
      ( rank( level ) >= rank( 'warn' ) ? process.stderr : process.stdout ).write( line );
    }
  };
  const logger = Object.fromEntries( LEVELS.map( ( level ) => [ level, ( message ) => write( level, message ) ] ) );
  logger.file = logFile;
  logger.close = () => fs.closeSync( fd );
  return logger;
}
