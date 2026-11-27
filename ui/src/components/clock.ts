// A time in a video as the player shows it: m:ss, or h:mm:ss past an hour.
export function clock( seconds: number ): string {
  const whole = Math.max( 0, Math.floor( seconds ) );
  const hours = Math.floor( whole / 3600 );
  const minutes = Math.floor( ( whole % 3600 ) / 60 );
  const rest = String( whole % 60 ).padStart( 2, '0' );
  return hours > 0 ? [ String( hours ), String( minutes ).padStart( 2, '0' ), rest ].join( ':' ) : [ String( minutes ), rest ].join( ':' );
}
